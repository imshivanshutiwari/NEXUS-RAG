"""Hybrid retriever: BM25 sparse + pgvector dense fused via RRF."""

from typing import Dict, List

from ingestion.indexer import ScoredDocument
from retrieval.dense_retriever import DenseRetriever
from retrieval.sparse_retriever import SparseRetriever
from utils.config_loader import ConfigLoader
from utils.logger import get_logger

logger = get_logger(__name__)

_RRF_K = 60


class HybridRetriever:
    """
    Hybrid retrieval combining BM25 sparse and pgvector dense results
    using Reciprocal Rank Fusion (RRF).
    """

    def __init__(self) -> None:
        try:
            cfg = ConfigLoader("retrieval_config.yaml")
            self._dense_w: float = cfg.get("hybrid.dense_weight", 0.7)
            self._sparse_w: float = cfg.get("hybrid.sparse_weight", 0.3)
            self._rrf_k: int = cfg.get("hybrid.rrf_k", _RRF_K)
            self._top_k: int = cfg.get("hybrid.top_k", 20)
        except Exception:
            self._dense_w = 0.7
            self._sparse_w = 0.3
            self._rrf_k = _RRF_K
            self._top_k = 20
        self.dense_retriever = DenseRetriever()
        self.sparse_retriever = SparseRetriever()

    def retrieve(self, query: str, k: int = 20) -> List[ScoredDocument]:
        """
        Full hybrid retrieval:
        1. dense ANN search → top 2k candidates
        2. BM25 search → top 2k candidates
        3. RRF fusion → top k
        """
        dense_results = self.dense_retriever.search(query, k=k * 2)
        bm25_scores = self.sparse_retriever.get_scores(query)
        fused = self.rrf_fusion(dense_results, bm25_scores, k=k)
        logger.debug(
            "HybridRetriever: dense=%d bm25=%d fused=%d",
            len(dense_results),
            len(bm25_scores),
            len(fused),
        )
        return fused

    def rrf_fusion(
        self,
        dense: List[ScoredDocument],
        sparse: Dict[str, float],
        k: int = 20,
    ) -> List[ScoredDocument]:
        """Reciprocal Rank Fusion: score(d) = Σ 1/(k + rank_i(d))."""
        dense_rank: Dict[str, int] = {r.doc_id: i + 1 for i, r in enumerate(dense)}
        bm25_sorted = sorted(sparse.items(), key=lambda x: -x[1])
        sparse_rank: Dict[str, int] = {
            doc_id: i + 1 for i, (doc_id, _) in enumerate(bm25_sorted)
        }
        doc_map: Dict[str, ScoredDocument] = {r.doc_id: r for r in dense}

        all_ids = set(dense_rank.keys()) | set(sparse_rank.keys())
        fused_scores: List[tuple] = []
        for doc_id in all_ids:
            score = 0.0
            if doc_id in dense_rank:
                score += self._dense_w / (self._rrf_k + dense_rank[doc_id])
            if doc_id in sparse_rank:
                score += self._sparse_w / (self._rrf_k + sparse_rank[doc_id])
            fused_scores.append((doc_id, score))

        fused_scores.sort(key=lambda x: -x[1])
        results: List[ScoredDocument] = []
        for doc_id, score in fused_scores[:k]:
            base = doc_map.get(doc_id)
            if base:
                results.append(
                    ScoredDocument(
                        doc_id=base.doc_id,
                        content=base.content,
                        score=score,
                        source=base.source,
                        metadata=base.metadata,
                        chunk_index=base.chunk_index,
                    )
                )
        return results

    def update_weights(self, dense_w: float, sparse_w: float) -> None:
        """Dynamically adjust fusion weights based on query type."""
        self._dense_w = dense_w
        self._sparse_w = sparse_w
        logger.info(
            "HybridRetriever: weights updated dense=%.2f sparse=%.2f", dense_w, sparse_w
        )
