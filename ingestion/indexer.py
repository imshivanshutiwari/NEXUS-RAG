"""pgvector HNSW indexer for document embeddings."""

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from utils.logger import get_logger

logger = get_logger(__name__)

_DDL = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    content TEXT NOT NULL,
    embedding vector(1024),
    metadata JSONB,
    source VARCHAR(50),
    created_at TIMESTAMP DEFAULT NOW(),
    chunk_index INTEGER,
    doc_id VARCHAR(255)
);

CREATE INDEX IF NOT EXISTS documents_embedding_hnsw
    ON documents USING hnsw (embedding vector_cosine_ops)
    WITH (m=16, ef_construction=64);
"""


@dataclass
class ScoredDocument:
    doc_id: str
    content: str
    score: float
    source: str = ""
    metadata: dict = field(default_factory=dict)
    chunk_index: int = 0


class PGVectorIndexer:
    """Insert and search documents in PostgreSQL + pgvector with HNSW index."""

    def __init__(self) -> None:
        self._conn = None
        self._connect()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create_schema(self) -> None:
        """Create extension, table, and HNSW index if they don't exist."""
        with self._conn.cursor() as cur:
            cur.execute(_DDL)
        self._conn.commit()
        logger.info("PGVectorIndexer: schema ready.")

    def insert_documents(
        self,
        docs: List[Dict[str, Any]],
        embeddings: np.ndarray,
    ) -> None:
        """Bulk-insert documents with their embeddings."""
        assert len(docs) == len(embeddings), "docs and embeddings length mismatch"
        with self._conn.cursor() as cur:
            for doc, emb in zip(docs, embeddings):
                cur.execute(
                    """
                    INSERT INTO documents (content, embedding, metadata, source, chunk_index, doc_id)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (
                        doc.get("content", ""),
                        emb.tolist(),
                        json.dumps(doc.get("metadata", {})),
                        doc.get("source", ""),
                        doc.get("chunk_index", 0),
                        doc.get("doc_id", ""),
                    ),
                )
        self._conn.commit()
        logger.info("PGVectorIndexer: inserted %d documents.", len(docs))

    def similarity_search(
        self,
        query_vec: np.ndarray,
        k: int = 20,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[ScoredDocument]:
        """ANN cosine similarity search. Returns top-k results."""
        with self._conn.cursor() as cur:
            base_sql = """
                SELECT doc_id, content, source, metadata, chunk_index,
                       1 - (embedding <=> %s::vector) AS score
                FROM documents
                {where}
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """
            vec_list = query_vec.tolist()
            if filter_metadata:
                where = "WHERE metadata @> %s::jsonb"
                cur.execute(
                    base_sql.format(where=where),
                    (vec_list, json.dumps(filter_metadata), vec_list, k),
                )
            else:
                cur.execute(base_sql.format(where=""), (vec_list, vec_list, k))

            rows = cur.fetchall()
        return [
            ScoredDocument(
                doc_id=row[0] or "",
                content=row[1],
                source=row[2] or "",
                metadata=row[3] or {},
                chunk_index=row[4] or 0,
                score=float(row[5]),
            )
            for row in rows
        ]

    def hybrid_search(
        self,
        query_vec: np.ndarray,
        bm25_scores: Dict[str, float],
        k: int = 10,
    ) -> List[ScoredDocument]:
        """RRF fusion of dense results + BM25 scores."""
        dense_results = self.similarity_search(query_vec, k=k * 2)
        return _rrf_fusion(dense_results, bm25_scores, k=k)

    def count(self) -> int:
        """Return total number of indexed documents."""
        with self._conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM documents")
            return cur.fetchone()[0]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _connect(self) -> None:
        try:
            import psycopg2

            self._conn = psycopg2.connect(
                host=os.getenv("POSTGRES_HOST", "localhost"),
                port=int(os.getenv("POSTGRES_PORT", "5432")),
                dbname=os.getenv("POSTGRES_DB", "nexusrag"),
                user=os.getenv("POSTGRES_USER", "nexus"),
                password=os.getenv("POSTGRES_PASSWORD", "nexuspass"),
            )
            logger.info("PGVectorIndexer: connected to PostgreSQL.")
        except Exception as exc:
            logger.error("PGVectorIndexer: connection failed: %s", exc)
            self._conn = None


def _rrf_fusion(
    dense_results: List[ScoredDocument],
    bm25_scores: Dict[str, float],
    k: int = 10,
    rrf_k: int = 60,
) -> List[ScoredDocument]:
    """Reciprocal Rank Fusion of dense + sparse results."""
    # Build rank maps
    dense_rank: Dict[str, int] = {r.doc_id: i + 1 for i, r in enumerate(dense_results)}
    bm25_sorted = sorted(bm25_scores.items(), key=lambda x: -x[1])
    sparse_rank: Dict[str, int] = {
        doc_id: i + 1 for i, (doc_id, _) in enumerate(bm25_sorted)
    }

    all_ids = set(dense_rank.keys()) | set(sparse_rank.keys())
    fused: List[tuple] = []
    doc_map: Dict[str, ScoredDocument] = {r.doc_id: r for r in dense_results}

    for doc_id in all_ids:
        rrf_score = 0.0
        if doc_id in dense_rank:
            rrf_score += 1.0 / (rrf_k + dense_rank[doc_id])
        if doc_id in sparse_rank:
            rrf_score += 1.0 / (rrf_k + sparse_rank[doc_id])
        fused.append((doc_id, rrf_score))

    fused.sort(key=lambda x: -x[1])
    results: List[ScoredDocument] = []
    for doc_id, score in fused[:k]:
        if doc_id in doc_map:
            r = doc_map[doc_id]
            results.append(
                ScoredDocument(
                    doc_id=r.doc_id,
                    content=r.content,
                    score=score,
                    source=r.source,
                    metadata=r.metadata,
                    chunk_index=r.chunk_index,
                )
            )
    return results
