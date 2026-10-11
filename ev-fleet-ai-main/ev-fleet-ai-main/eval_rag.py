"""Evaluate the Transcil RAG pipeline with RAGAS using Gemini as the judge."""

from __future__ import annotations

import logging
import os
import sys
import time
import types
from typing import Any

# RAGAS 0.4 still imports Vertex AI chat models from a langchain-community path
# that was removed. Stub the module so LangchainLLMWrapper can load.
_vertex_mod = "langchain_community.chat_models.vertexai"
if _vertex_mod not in sys.modules:
    try:
        __import__(_vertex_mod)
    except ModuleNotFoundError:
        _stub = types.ModuleType(_vertex_mod)
        _stub.ChatVertexAI = type("ChatVertexAI", (), {})  # type: ignore[attr-defined]
        sys.modules[_vertex_mod] = _stub

from datasets import Dataset
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from agents import AppConfig, _plain_text, rag_node

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("eval_rag")

GEMINI_MODEL = "gemini-3.6-flash"
EMBEDDING_MODEL = "models/gemini-embedding-001"
RESULTS_CSV = "rag_eval_results.csv"

GOLDEN_QA: list[dict[str, str]] = [
    {
        "question": "Who is the corporate parent of Transcil?",
        "ground_truth": "Transcil operates under the Cal-On Group.",
    },
    {
        "question": "What are the battery specifications for ELACIL 2.5?",
        "ground_truth": "1.8-3.6 kWh LFP A-Grade Battery.",
    },
    {
        "question": "What is Transcil's target service resolution rate?",
        "ground_truth": (
            "Most service issues are resolved within 24 hours (90% resolution rate)."
        ),
    },
    {
        "question": "What is included in every Transcil pricing plan?",
        "ground_truth": (
            "Battery swapping, 24/7 battery delivery, 24/7 on-road assistance, "
            "unlimited kilometers, and servicing."
        ),
    },
]


def _config_from_env() -> AppConfig:
    api_key = os.getenv("GOOGLE_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is missing. Set it in the .env file.")
    return {
        "google_api_key": api_key,
        "mysql_host": os.getenv("MYSQL_HOST", "localhost"),
        "mysql_port": os.getenv("MYSQL_PORT", "3306"),
        "mysql_user": os.getenv("MYSQL_USER", "root"),
        "mysql_password": os.getenv("MYSQL_PASSWORD", ""),
        "mysql_database": os.getenv("MYSQL_DATABASE", "ev_fleet"),
    }


def _build_judge(api_key: str) -> tuple[Any, Any]:
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper

    llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=api_key,
        temperature=0,
    )
    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=api_key,
    )
    return LangchainLLMWrapper(llm), LangchainEmbeddingsWrapper(embeddings)


def _run_rag_samples(config: AppConfig) -> Dataset:
    questions: list[str] = []
    answers: list[str] = []
    contexts: list[list[str]] = []
    ground_truths: list[str] = []

    logger.info("Running RAG pipeline on %s golden questions", len(GOLDEN_QA))
    for index, item in enumerate(GOLDEN_QA, start=1):
        question = item["question"]
        logger.info("[%s/%s] %s", index, len(GOLDEN_QA), question)
        try:
            result = rag_node({"question": question}, config)
        except Exception:
            logger.exception("RAG node failed for question %s", index)
            result = {
                "final_answer": "",
                "retrieved_docs": [],
            }

        answer = _plain_text(result.get("final_answer"))
        retrieved = result.get("retrieved_docs") or []
        docs = [str(doc) for doc in retrieved if str(doc).strip()]
        if not docs:
            logger.warning("No retrieved contexts for question %s", index)
            docs = [""]

        questions.append(question)
        answers.append(answer)
        contexts.append(docs)
        ground_truths.append(item["ground_truth"])
        logger.info("Answer: %s", answer[:240] + ("..." if len(answer) > 240 else ""))
        time.sleep(1.5)

    return Dataset.from_dict(
        {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        }
    )


def _to_ragas_dataset(dataset: Dataset) -> Any:
    """Prefer HuggingFace columns; map to RAGAS EvaluationDataset if required."""
    try:
        from ragas import EvaluationDataset

        rows = []
        for row in dataset:
            rows.append(
                {
                    "user_input": row["question"],
                    "response": row["answer"],
                    "retrieved_contexts": row["contexts"],
                    "reference": row["ground_truth"],
                }
            )
        return EvaluationDataset.from_list(rows)
    except Exception:
        logger.info("Using HuggingFace Dataset columns for RAGAS evaluate().")
        return dataset


def _load_metrics(evaluator_llm: Any, evaluator_embeddings: Any) -> list[Any]:
    try:
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )

        return [
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ]
    except ImportError:
        from ragas.metrics import (
            AnswerRelevancy,
            Faithfulness,
            LLMContextPrecisionWithReference,
            LLMContextRecall,
        )

        return [
            Faithfulness(llm=evaluator_llm),
            AnswerRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings),
            LLMContextPrecisionWithReference(llm=evaluator_llm),
            LLMContextRecall(llm=evaluator_llm),
        ]


def _metric_column_names(frame: Any) -> list[str]:
    known = {
        "faithfulness",
        "answer_relevancy",
        "answer_relevance",
        "context_precision",
        "llm_context_precision_with_reference",
        "context_recall",
        "llm_context_recall",
        "nv_context_recall",
    }
    skip = {
        "question",
        "answer",
        "contexts",
        "ground_truth",
        "user_input",
        "response",
        "retrieved_contexts",
        "reference",
    }
    names = []
    for col in frame.columns:
        if col in skip:
            continue
        if col in known or col.endswith("_score"):
            names.append(col)
    return names or [col for col in frame.columns if col not in skip]


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if hasattr(value, "value"):
        return _as_float(value.value)
    if isinstance(value, (list, tuple)):
        nums = [_as_float(item) for item in value]
        nums = [item for item in nums if item is not None]
        return sum(nums) / len(nums) if nums else None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _print_summary(frame: Any) -> None:
    metric_cols = _metric_column_names(frame)
    if not metric_cols:
        logger.warning("No metric columns found in results: %s", list(frame.columns))
        print(frame)
        return

    print("\n=== RAGAS overall averages ===")
    header = f"{'metric':<40} {'score':>8}"
    print(header)
    print("-" * len(header))
    for col in metric_cols:
        scores = [_as_float(value) for value in frame[col].tolist()]
        scores = [score for score in scores if score is not None]
        avg = sum(scores) / len(scores) if scores else float("nan")
        print(f"{col:<40} {avg:>8.4f}")

    print("\n=== Per-question breakdown ===")
    question_col = "question" if "question" in frame.columns else "user_input"
    for index, row in frame.iterrows():
        question = str(row.get(question_col, f"row_{index}"))
        print(f"\nQ{index + 1}: {question}")
        for col in metric_cols:
            value = _as_float(row.get(col))
            if value is None:
                print(f"  {col}: {row.get(col)}")
            else:
                print(f"  {col}: {value:.4f}")


def main() -> int:
    try:
        config = _config_from_env()
        evaluator_llm, evaluator_embeddings = _build_judge(config["google_api_key"])
        hf_dataset = _run_rag_samples(config)
        eval_dataset = _to_ragas_dataset(hf_dataset)
        metrics = _load_metrics(evaluator_llm, evaluator_embeddings)

        try:
            import nest_asyncio

            nest_asyncio.apply()
        except Exception:
            logger.debug("nest_asyncio not applied")

        from ragas import evaluate
        from ragas.run_config import RunConfig

        logger.info("Evaluating with RAGAS metrics: faithfulness, answer_relevancy, context_precision, context_recall")
        result = evaluate(
            dataset=eval_dataset,
            metrics=metrics,
            llm=evaluator_llm,
            embeddings=evaluator_embeddings,
            raise_exceptions=False,
            batch_size=1,
            run_config=RunConfig(max_workers=1, timeout=180),
        )
        frame = result.to_pandas()
        if "question" not in frame.columns:
            frame.insert(0, "question", list(hf_dataset["question"]))
        if "answer" not in frame.columns:
            frame.insert(1, "answer", list(hf_dataset["answer"]))
        if "ground_truth" not in frame.columns:
            frame.insert(2, "ground_truth", list(hf_dataset["ground_truth"]))

        _print_summary(frame)
        frame.to_csv(RESULTS_CSV, index=False)
        logger.info("Saved evaluation results to %s", RESULTS_CSV)
        return 0
    except Exception:
        logger.exception("RAG evaluation failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
