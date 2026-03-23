"""Dense retriever using pgvector ANN search."""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from ingestion.embedder import DocumentEmbedder
from ingestion.indexer import PGVectorIndexer, ScoredDocument
from utils.logger import get_logger

logger = get_logger(__name__)


class DenseRetriever:
    """Retrieve documents using dense vector similarity (pgvector HNSW)."""

    def __init__(self) -> None:
        self.embedder = DocumentEmbedder()
        self.indexer = PGVectorIndexer()

    def search(
        self,
        query: str,
        k: int = 20,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[ScoredDocument]:
        """Embed *query* and run ANN search returning top-k results."""
        query_vec = self.embedder.embed_query(query)
        results = self.indexer.similarity_search(query_vec, k=k, filter_metadata=filter_metadata)
        logger.debug("DenseRetriever: %d results for query='%s...'", len(results), query[:60])
        return results

    def search_by_vector(self, query_vec: np.ndarray, k: int = 20) -> List[ScoredDocument]:
        """Search directly by a pre-computed query vector."""
        return self.indexer.similarity_search(query_vec, k=k)
