"""LangGraph node functions: router, text-to-SQL, RAG, and general."""

from __future__ import annotations

import ast
import re
from typing import Any, Literal, TypedDict
from urllib.parse import quote_plus

from flashrank import Ranker  # noqa: F401  # required before FlashrankRerank
from langchain_classic.chains import create_sql_query_chain
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_community.document_compressors import FlashrankRerank
from langchain_core.prompts import PromptTemplate
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_community.utilities import SQLDatabase
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from document_loader import get_vector_store, is_vector_store_empty

GEMINI_MODEL = "gemini-3.6-flash"

Category = Literal["db", "doc", "general"]

ROUTER_PROMPT = """
You are a highly precise routing assistant for Transcil, an EV fleet operations company.
Your task is to classify the user's query into one of three distinct categories based on the required data source.
Reply with exactly one word: 'db', 'doc', or 'general'. Do not include punctuation or explanations.

Routing Logic:
1. 'db' (Transactional & Real-Time Data): Route here ONLY if the user is asking about specific, live database records. This includes queries about specific rider accounts, current vehicle statuses (e.g., Available, Maintenance), individual assigned vehicles, or active verification statuses. Examples: "Is vehicle ID 405 available?", "Show me the joined date for rider John Doe."
2. 'doc' (Company Knowledge & Offerings): Route here for ANY questions regarding Transcil's general information, services, or static product details. This includes company metrics, services offered, electric fleet models (ELACIL 2.5, ELLOD ELITE), general battery specifications, pricing plan structures (weekly/monthly rates), how to join, partnership programs, and sustainability impact. Examples: "What are your weekly pricing plans?", "What is the battery type for the ELACIL 2.5?", "How do I become a partner?"
3. 'general' (Conversational): Route here for basic greetings, pleasantries, or questions completely unrelated to Transcil, EVs, or logistics.
"""

SQL_SCHEMA = """
Database: `ev_fleet`
Table `vehicles`: `vehicle_id` INT PK, `model_name` VARCHAR(50), `battery_type` VARCHAR(50) DEFAULT '1.8-3.6 kWh LFP A-Grade', `range_per_charge_km` INT, `weekly_plan_rate` DECIMAL(8,2), `monthly_plan_rate` DECIMAL(8,2), `status` ENUM('Available', 'Rented', 'Maintenance', 'Charging'), `created_at` TIMESTAMP.
Table `riders`: `rider_id` INT PK, `full_name` VARCHAR(100), `phone_number` VARCHAR(15), `assigned_vehicle_id` INT FK -> vehicles.vehicle_id, `plan_type` ENUM('Weekly', 'Monthly'), `verification_status` ENUM('Pending', 'Verified', 'Rejected'), `joined_date` DATE.
Never execute DML statements (INSERT, UPDATE, DELETE).
"""

SQL_SYSTEM_PROMPT = (
    "You are a MySQL expert. Given an input question, create a syntactically "
    "correct MySQL query to run, then look at the results and return the answer. "
    f"Use the following schema:{SQL_SCHEMA}"
)

RAG_PROMPT = (
    "You are an assistant for question-answering tasks. Use the following retrieved "
    "document context to answer the question. If you don't know the answer, just "
    "say that you don't know. Keep the answer concise."
)

DML_PATTERN = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|TRUNCATE|CREATE|REPLACE|GRANT|REVOKE)\b",
    re.IGNORECASE,
)


class AppConfig(TypedDict):
    google_api_key: str
    mysql_host: str
    mysql_port: str
    mysql_user: str
    mysql_password: str
    mysql_database: str


def get_llm(google_api_key: str) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=google_api_key,
        temperature=0,
    )


def _plain_text(value: Any) -> str:
    """Extract a clean string from Gemini/LangChain message content."""
    if value is None:
        return ""
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("[{") and "text" in stripped:
            try:
                return _plain_text(ast.literal_eval(stripped))
            except (ValueError, SyntaxError):
                return stripped
        return stripped
    content = getattr(value, "content", None)
    if content is not None and content is not value:
        text = _plain_text(content)
        if text:
            return text
    text_attr = getattr(value, "text", None)
    if isinstance(text_attr, str) and text_attr.strip():
        return text_attr.strip()
    if callable(text_attr):
        try:
            text = text_attr()
            if isinstance(text, str) and text.strip():
                return text.strip()
        except TypeError:
            pass
    if isinstance(value, dict):
        if value.get("type") == "text" or "text" in value:
            return _plain_text(value.get("text", ""))
        return _plain_text(value.get("content"))
    if isinstance(value, (list, tuple)):
        parts = [_plain_text(item) for item in value]
        return "\n".join(part for part in parts if part).strip()
    return str(value).strip()


def _mysql_uri(config: AppConfig) -> str:
    user = quote_plus(config["mysql_user"])
    password = quote_plus(config["mysql_password"])
    host = config["mysql_host"]
    port = config["mysql_port"]
    database = config["mysql_database"]
    return f"mysql+mysqlconnector://{user}:{password}@{host}:{port}/{database}"


def _clean_sql(raw: str) -> str:
    text = raw.strip()
    if "```" in text:
        match = re.search(r"```(?:sql)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
        if match:
            text = match.group(1).strip()
    for prefix in ("SQLQuery:", "SQL Query:", "SQL:", "Query:"):
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix) :].strip()
    return text.strip()


def _parse_category(raw: str) -> Category:
    token = re.sub(r"[^a-z]", " ", raw.strip().lower()).split()
    if not token:
        return "general"
    first = token[0]
    if first in ("db", "doc", "general"):
        return first  # type: ignore[return-value]
    if "db" in token or "database" in token or "sql" in token:
        return "db"
    if "doc" in token or "document" in token:
        return "doc"
    if "general" in token:
        return "general"
    return "general"


def router_node(state: dict[str, Any], config: AppConfig) -> dict[str, Any]:
    llm = get_llm(config["google_api_key"])
    response = llm.invoke(
        [
            SystemMessage(content=ROUTER_PROMPT),
            HumanMessage(content=state["question"]),
        ]
    )
    category = _parse_category(_plain_text(response))
    return {"category": category}


def text_to_sql_node(state: dict[str, Any], config: AppConfig) -> dict[str, Any]:
    question = state["question"]
    try:
        db = SQLDatabase.from_uri(_mysql_uri(config))
    except Exception as exc:
        return {
            "sql_query": "",
            "sql_result": "",
            "final_answer": (
                "Could not connect to MySQL. Check MYSQL_* values in the .env file "
                f"and that the `ev_fleet` database exists. Details: {exc}"
            ),
        }

    prompt = PromptTemplate(
        input_variables=["input", "table_info", "top_k", "dialect"],
        template=(
            SQL_SYSTEM_PROMPT
            + "\n\nDialect: {dialect}\nOnly use tables that exist.\n"
            "Table info:\n{table_info}\n"
            "Return at most {top_k} rows.\n"
            "Question: {input}\nSQL query:"
        ),
    )

    try:
        query_chain = create_sql_query_chain(get_llm(config["google_api_key"]), db, prompt)
        raw_sql = query_chain.invoke({"question": question})
        sql_query = _clean_sql(_plain_text(raw_sql))
        if not sql_query or DML_PATTERN.search(sql_query):
            return {
                "sql_query": sql_query,
                "sql_result": "",
                "final_answer": (
                    "That request would require a modifying SQL statement, which is "
                    "not allowed. Ask a read-only question about riders or vehicles."
                ),
            }

        execute_tool = QuerySQLDataBaseTool(db=db)
        sql_result = execute_tool.invoke({"query": sql_query})

        synthesizer = get_llm(config["google_api_key"])
        synthesis = synthesizer.invoke(
            [
                SystemMessage(
                    content=(
                        "You are a helpful EV fleet assistant. Answer the user in "
                        "natural language using only the SQL results. Be concise."
                    )
                ),
                HumanMessage(
                    content=(
                        f"Question: {question}\nSQL: {sql_query}\n"
                        f"Result: {sql_result}"
                    )
                ),
            ]
        )
        return {
            "sql_query": sql_query,
            "sql_result": str(sql_result),
            "final_answer": _plain_text(synthesis),
        }
    except Exception as exc:
        return {
            "sql_query": "",
            "sql_result": "",
            "final_answer": f"Text-to-SQL failed: {exc}",
        }


def rag_node(state: dict[str, Any], config: AppConfig) -> dict[str, Any]:
    question = state["question"]
    api_key = config["google_api_key"]

    if is_vector_store_empty(api_key):
        return {
            "retrieved_docs": [],
            "final_answer": (
                "The document store is empty. Upload a .txt or .pdf company "
                "document in the sidebar, then ask again."
            ),
        }

    try:
        store = get_vector_store(api_key)
        base_retriever = store.as_retriever(search_kwargs={"k": 8})
        compressor = FlashrankRerank(top_n=3)
        retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=base_retriever,
        )
        docs = retriever.invoke(question)
        if not docs:
            return {
                "retrieved_docs": [],
                "final_answer": (
                    "I could not find relevant passages in the uploaded documents "
                    "for that question."
                ),
            }

        context = "\n\n".join(doc.page_content for doc in docs)
        llm = get_llm(api_key)
        response = llm.invoke(
            [
                SystemMessage(content=RAG_PROMPT),
                HumanMessage(content=f"Context:\n{context}\n\nQuestion: {question}"),
            ]
        )
        return {
            "retrieved_docs": [doc.page_content for doc in docs],
            "final_answer": _plain_text(response),
        }
    except Exception as exc:
        return {
            "retrieved_docs": [],
            "final_answer": f"Document search failed: {exc}",
        }


def general_node(state: dict[str, Any], config: AppConfig) -> dict[str, Any]:
    llm = get_llm(config["google_api_key"])
    response = llm.invoke(
        [
            SystemMessage(
                content=(
                    "You are a helpful assistant for an EV fleet company. "
                    "Answer clearly and concisely."
                )
            ),
            HumanMessage(content=state["question"]),
        ]
    )
    return {"final_answer": _plain_text(response)}
