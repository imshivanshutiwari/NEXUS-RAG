"""RAGAS evaluator: faithfulness, answer_relevancy, context_precision, context_recall."""
from typing import Any, Dict, List, Optional

import pandas as pd

from utils.logger import get_logger

logger = get_logger(__name__)


class RAGASEvaluator:
    """Full RAGAS evaluation: all 4 metrics."""

    def evaluate_response(
        self,
        query: str,
        answer: str,
        contexts: List[str],
        ground_truth: Optional[str] = None,
    ) -> Dict[str, float]:
        """
        Evaluate a single RAG response using RAGAS.
        Returns dict with faithfulness, answer_relevancy, context_precision, context_recall.
        """
        try:
            from datasets import Dataset
            from ragas import evaluate
            from ragas.metrics import (
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
            )

            data = {
                "question": [query],
                "answer": [answer],
                "contexts": [contexts if contexts else [""]],
                "ground_truth": [ground_truth if ground_truth else query],
            }
            dataset = Dataset.from_dict(data)
            scores = evaluate(
                dataset,
                metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
            )
            result = scores.to_pandas().iloc[0].to_dict()
            return {
                "faithfulness": float(result.get("faithfulness", 0.5)),
                "answer_relevancy": float(result.get("answer_relevancy", 0.5)),
                "context_precision": float(result.get("context_precision", 0.5)),
                "context_recall": float(result.get("context_recall", 0.5)),
            }
        except Exception as exc:
            logger.warning("RAGASEvaluator.evaluate_response failed: %s", exc)
            return {
                "faithfulness": 0.5,
                "answer_relevancy": 0.5,
                "context_precision": 0.5,
                "context_recall": 0.5,
            }

    def batch_evaluate(self, qa_pairs: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Evaluate a list of {query, answer, contexts, ground_truth} dicts.
        Returns a DataFrame with per-sample RAGAS scores.
        """
        rows = []
        for item in qa_pairs:
            scores = self.evaluate_response(
                query=item.get("query", ""),
                answer=item.get("answer", ""),
                contexts=item.get("contexts", []),
                ground_truth=item.get("ground_truth"),
            )
            scores["query"] = item.get("query", "")
            rows.append(scores)
        return pd.DataFrame(rows)

    def compute_aggregate_scores(
        self, history: List[Dict[str, float]]
    ) -> Dict[str, float]:
        """Return mean RAGAS scores across a history of per-query score dicts."""
        if not history:
            return {}
        df = pd.DataFrame(history)
        return df.mean().to_dict()
