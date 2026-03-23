"""Reranker agent: CrossEncoder → top 10, ColBERT → top 5."""
from typing import Any, List

from agents.state import RAGState
from utils.logger import get_logger

logger = get_logger(__name__)

_RELEVANCE_THRESHOLD = 0.0


def reranker_node(state: RAGState) -> RAGState:
    """
    LangGraph node: Reranking.
    1. CrossEncoder reranking → top 10
    2. ColBERT late interaction → top 5
    3. Filter by relevance threshold
    """
    from ingestion.indexer import ScoredDocument
    from retrieval.reranker import CrossEncoderReranker
    from retrieval.colbert_reranker import ColBERTReranker

    query = state["query"]
    retrieved = state.get("retrieved_docs", [])

    if not retrieved:
        state["reranked_docs"] = []
        state["pipeline_trace"] = state.get("pipeline_trace", []) + ["reranker: no docs to rerank"]
        return state

    # Build ScoredDocument objects
    scored_docs = [
        ScoredDocument(
            doc_id=d["doc_id"],
            content=d.get("content", ""),
            score=d.get("score", 0.0),
            source=d.get("source", ""),
            metadata=d.get("metadata", {}),
            chunk_index=d.get("chunk_index", 0),
        )
        for d in retrieved
    ]

    # CrossEncoder → top 10
    ce_reranker = CrossEncoderReranker()
    ce_top10 = ce_reranker.rerank(query, scored_docs, top_k=10)

    # ColBERT → top 5
    colbert = ColBERTReranker()
    final_top5 = colbert.rerank(query, ce_top10, top_k=5)

    state["reranked_docs"] = [
        {
            "doc_id": d.doc_id,
            "content": d.content,
            "score": d.score,
            "source": d.source,
            "metadata": d.metadata,
            "chunk_index": d.chunk_index,
        }
        for d in final_top5
    ]
    state["pipeline_trace"] = state.get("pipeline_trace", []) + [
        f"reranker: {len(retrieved)}→CE10→ColBERT{len(final_top5)}"
    ]
    logger.info("RerankerAgent: final %d docs.", len(final_top5))
    return state
