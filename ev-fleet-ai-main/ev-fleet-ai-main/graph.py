"""LangGraph state and compiled EV fleet workflow."""

from __future__ import annotations

from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from agents import AppConfig, general_node, rag_node, router_node, text_to_sql_node


class GraphState(TypedDict, total=False):
    question: str
    category: Literal["db", "doc", "general"]
    sql_query: str
    sql_result: str
    retrieved_docs: list[str]
    final_answer: str


def _route(state: GraphState) -> Literal["text_to_sql", "rag", "general"]:
    mapping = {"db": "text_to_sql", "doc": "rag", "general": "general"}
    return mapping.get(state.get("category", "general"), "general")  # type: ignore[return-value]


def build_graph(config: AppConfig) -> Any:
    workflow = StateGraph(GraphState)

    workflow.add_node("router", lambda state: router_node(state, config))
    workflow.add_node("text_to_sql", lambda state: text_to_sql_node(state, config))
    workflow.add_node("rag", lambda state: rag_node(state, config))
    workflow.add_node("general", lambda state: general_node(state, config))

    workflow.add_edge(START, "router")
    workflow.add_conditional_edges(
        "router",
        _route,
        {
            "text_to_sql": "text_to_sql",
            "rag": "rag",
            "general": "general",
        },
    )
    workflow.add_edge("text_to_sql", END)
    workflow.add_edge("rag", END)
    workflow.add_edge("general", END)

    return workflow.compile()
