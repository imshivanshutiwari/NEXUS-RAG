"""ColBERT late interaction reranker: MaxSim scoring."""
from dataclasses import dataclass, field
from typing import List

import numpy as np

from ingestion.indexer import ScoredDocument
from utils.logger import get_logger

logger = get_logger(__name__)


class ColBERTReranker:
    """
    Late interaction reranking using ColBERT MaxSim scoring.

    ColBERT MaxSim:  score(q, d) = Σ_qi  max_dj  cos(E_q(qi), E_d(dj))
    """

    def __init__(self) -> None:
        self._model = None

    def _get_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer

                # Use a token-level model as ColBERT proxy
                self._model = SentenceTransformer("multi-qa-mpnet-base-v2")
                logger.info("ColBERTReranker: loaded SentenceTransformer as ColBERT proxy.")
            except Exception as exc:
                logger.error("ColBERTReranker: model load failed: %s", exc)
        return self._model

    def rerank(
        self,
        query: str,
        documents: List[ScoredDocument],
        top_k: int = 5,
    ) -> List[ScoredDocument]:
        """Rerank documents via MaxSim late interaction."""
        if not documents:
            return []
        model = self._get_model()
        if model is None:
            return documents[:top_k]

        # Tokenize and encode at word level (proxy for token embeddings)
        q_tokens = query.lower().split()
        q_embeddings = model.encode(q_tokens, normalize_embeddings=True)  # (Lq, D)

        scored: List[tuple] = []
        for doc in documents:
            d_tokens = doc.content.lower().split()[:128]  # cap doc length
            if not d_tokens:
                scored.append((doc, 0.0))
                continue
            d_embeddings = model.encode(d_tokens, normalize_embeddings=True)  # (Ld, D)
            score = self._maxsim(q_embeddings, d_embeddings)
            scored.append((doc, score))

        scored.sort(key=lambda x: -x[1])
        return [
            ScoredDocument(
                doc_id=d.doc_id,
                content=d.content,
                score=float(s),
                source=d.source,
                metadata=d.metadata,
                chunk_index=d.chunk_index,
            )
            for d, s in scored[:top_k]
        ]

    def batch_rerank(
        self,
        queries: List[str],
        doc_lists: List[List[ScoredDocument]],
        top_k: int = 5,
    ) -> List[List[ScoredDocument]]:
        """Rerank multiple query–document list pairs."""
        return [self.rerank(q, docs, top_k) for q, docs in zip(queries, doc_lists)]

    @staticmethod
    def _maxsim(q_emb: np.ndarray, d_emb: np.ndarray) -> float:
        """Compute ColBERT MaxSim: Σ_qi max_dj cos(q_i, d_j)."""
        # q_emb: (Lq, D),  d_emb: (Ld, D)
        sim_matrix = q_emb @ d_emb.T  # (Lq, Ld)
        return float(sim_matrix.max(axis=1).sum())
