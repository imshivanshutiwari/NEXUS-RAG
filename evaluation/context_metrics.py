"""Context precision and recall metrics for RAG evaluation."""

from typing import Dict, List

import numpy as np

from utils.logger import get_logger

logger = get_logger(__name__)


class ContextMetrics:
    """
    Compute context precision and context recall.

    - Context Precision: fraction of retrieved contexts that are relevant to the query.
    - Context Recall: fraction of ground-truth information covered by retrieved contexts.
    """

    def __init__(self) -> None:
        self._embedder = None

    def compute_precision(
        self, query: str, contexts: List[str], threshold: float = 0.4
    ) -> Dict[str, float]:
        """
        Compute context precision as mean relevance of retrieved contexts to *query*.
        Uses cosine similarity between query embedding and each context embedding.
        """
        if not contexts:
            return {"context_precision": 0.0}
        embedder = self._get_embedder()
        try:
            q_emb = embedder.embed_query(query)
            c_embs = embedder.embed_documents(contexts)
            sims = [
                float(embedder.compute_similarity(q_emb, c_embs[i]))
                for i in range(len(contexts))
            ]
            relevant = [s for s in sims if s >= threshold]
            precision = len(relevant) / len(sims) if sims else 0.0
            return {"context_precision": max(0.0, min(1.0, precision))}
        except Exception as exc:
            logger.warning("ContextMetrics.compute_precision failed: %s", exc)
            return {"context_precision": 0.5}

    def compute_recall(
        self,
        ground_truth: str,
        contexts: List[str],
        threshold: float = 0.4,
    ) -> Dict[str, float]:
        """
        Compute context recall as how well the contexts cover the ground truth.
        Approximated by cosine similarity between ground truth embedding and
        the mean of context embeddings.
        """
        if not contexts or not ground_truth:
            return {"context_recall": 0.0}
        embedder = self._get_embedder()
        try:
            gt_emb = embedder.embed_query(ground_truth)
            c_embs = embedder.embed_documents(contexts)
            mean_c_emb = np.mean(c_embs, axis=0)
            recall = float(embedder.compute_similarity(gt_emb, mean_c_emb))
            return {"context_recall": max(0.0, min(1.0, recall))}
        except Exception as exc:
            logger.warning("ContextMetrics.compute_recall failed: %s", exc)
            return {"context_recall": 0.5}

    def _get_embedder(self):
        if self._embedder is None:
            from ingestion.embedder import DocumentEmbedder

            self._embedder = DocumentEmbedder()
        return self._embedder
