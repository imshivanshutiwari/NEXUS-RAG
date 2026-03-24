"""Full benchmark pipeline: evaluate NEXUS-RAG on a real question set."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from evaluation.ragas_evaluator import RAGASEvaluator
from utils.logger import get_logger

logger = get_logger(__name__)

_RESULTS_DIR = Path("assets/results")

_BENCHMARK_QUESTIONS = [
    "What is Retrieval-Augmented Generation (RAG)?",
    "How does pgvector implement HNSW indexing?",
    "What are the key differences between BM25 and dense retrieval?",
    "Explain how LangGraph manages state in a multi-agent pipeline.",
    "What is the ColBERT late interaction model?",
    "How does Evidently AI detect embedding drift?",
    "What are RAGAS metrics and how are they computed?",
    "What is the role of cross-encoder reranking in RAG pipelines?",
    "How does HyDE (Hypothetical Document Embeddings) improve retrieval?",
    "What are the main advantages of hybrid search over pure dense retrieval?",
]


class BenchmarkRunner:
    """Run full end-to-end benchmark evaluation."""

    def __init__(self) -> None:
        _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        self.evaluator = RAGASEvaluator()

    def run(
        self,
        questions: Optional[List[str]] = None,
        output_path: Optional[Path] = None,
    ) -> pd.DataFrame:
        """
        Run the benchmark pipeline:
        1. For each question, run the full NEXUS-RAG graph
        2. Evaluate with RAGAS
        3. Save results to CSV
        """
        from agents.graph import NEXUSRAGGraph

        rag = NEXUSRAGGraph()
        questions = questions or _BENCHMARK_QUESTIONS
        rows: List[Dict[str, Any]] = []

        for i, question in enumerate(questions):
            logger.info(
                "BenchmarkRunner: Q%d/%d: '%s...'", i + 1, len(questions), question[:60]
            )
            try:
                state = rag.run(question)
                answer = state.get("final_answer", "")
                contexts = [d["content"] for d in state.get("reranked_docs", [])]
                scores = self.evaluator.evaluate_response(
                    query=question,
                    answer=answer,
                    contexts=contexts,
                    ground_truth=question,
                )
                row = {
                    "question": question,
                    "answer": answer[:500],
                    "healing_attempts": state.get("healing_attempts", 0),
                    "query_type": state.get("query_type", ""),
                    **scores,
                }
            except Exception as exc:
                logger.error("BenchmarkRunner: failed on Q%d: %s", i + 1, exc)
                row = {
                    "question": question,
                    "answer": "",
                    "healing_attempts": 0,
                    "query_type": "unknown",
                    "faithfulness": 0.0,
                    "answer_relevancy": 0.0,
                    "context_precision": 0.0,
                    "context_recall": 0.0,
                    "error": str(exc),
                }
            rows.append(row)

        df = pd.DataFrame(rows)
        output = output_path or _RESULTS_DIR / "benchmark_results.csv"
        df.to_csv(output, index=False)
        logger.info("BenchmarkRunner: saved results to %s", output)

        # Print summary
        numeric_cols = [
            "faithfulness",
            "answer_relevancy",
            "context_precision",
            "context_recall",
        ]
        for col in numeric_cols:
            if col in df.columns:
                logger.info("  Mean %s: %.3f", col, df[col].mean())

        return df

    def compare_with_baseline(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Compare NEXUS-RAG scores vs naive RAG baseline (dense-only, no reranking)."""
        from retrieval.dense_retriever import DenseRetriever
        from generation.bedrock_client import BedrockClient
        from generation.prompt_builder import PromptBuilder

        baseline_rows = []
        for _, row in df.iterrows():
            question = row["question"]
            try:
                docs = DenseRetriever().search(question, k=5)
                contexts = [d.content for d in docs]
                prompt = PromptBuilder().build(question, contexts)
                answer = BedrockClient().generate(prompt)
                scores = self.evaluator.evaluate_response(
                    query=question, answer=answer, contexts=contexts
                )
                baseline_rows.append(scores)
            except Exception:
                baseline_rows.append(
                    {
                        "faithfulness": 0.0,
                        "answer_relevancy": 0.0,
                        "context_precision": 0.0,
                        "context_recall": 0.0,
                    }
                )

        baseline_df = pd.DataFrame(baseline_rows)
        return {
            "nexus_rag_mean": df[["faithfulness", "answer_relevancy"]].mean().to_dict(),
            "baseline_mean": baseline_df[["faithfulness", "answer_relevancy"]]
            .mean()
            .to_dict(),
        }


if __name__ == "__main__":
    runner = BenchmarkRunner()
    results = runner.run()
    print(results.to_string())
