"""Sparse (BM25) retriever."""

from dataclasses import dataclass, field
from typing import List, Tuple

from ingestion.bm25_indexer import BM25Indexer
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ScoredDocument:
    doc_id: str
    content: str = ""
    score: float = 0.0
    source: str = ""
    metadata: dict = field(default_factory=dict)
    chunk_index: int = 0


class SparseRetriever:
    """BM25-based sparse retriever."""

    def __init__(self) -> None:
        self.bm25 = BM25Indexer()

    def search(self, query: str, k: int = 40) -> List[ScoredDocument]:
        """Return top-k BM25-scored documents for *query*."""
        results = self.bm25.search(query, k=k)
        return [ScoredDocument(doc_id=doc_id, score=score) for doc_id, score in results]

    def get_scores(self, query: str) -> dict:
        """Return {doc_id: bm25_score} for all indexed docs."""
        return self.bm25.get_scores_dict(query)
