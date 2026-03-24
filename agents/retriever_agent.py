"""Retriever agent: orchestrates hybrid retrieval + query expansion."""

from typing import Dict

from agents.state import RAGState
from utils.logger import get_logger

logger = get_logger(__name__)


def retriever_node(state: RAGState) -> RAGState:
    """
    LangGraph node: Retrieval.
    Runs hybrid BM25 + pgvector HNSW retrieval with HyDE + multi-query expansion.
    Collects top-20 candidates.
    """
    from retrieval.query_expander import QueryExpander

    query = state["query"]
    query_type = state.get("query_type", "factual")

    # Adjust retrieval weights based on query type
    from retrieval.hybrid_retriever import HybridRetriever

    retriever = HybridRetriever()
    if query_type == "factual":
        retriever.update_weights(dense_w=0.7, sparse_w=0.3)
    elif query_type == "analytical":
        retriever.update_weights(dense_w=0.8, sparse_w=0.2)
    elif query_type == "comparative":
        retriever.update_weights(dense_w=0.6, sparse_w=0.4)

    # Expand and retrieve
    expander = QueryExpander()
    expanded_queries = expander.multi_query_expand(query, n=3)
    state["expanded_queries"] = [query] + expanded_queries

    # Collect results from all expanded queries
    all_docs: Dict[str, dict] = {}
    for q in state["expanded_queries"]:
        docs = retriever.retrieve(q, k=20)
        for d in docs:
            if d.doc_id not in all_docs:
                all_docs[d.doc_id] = {
                    "doc_id": d.doc_id,
                    "content": d.content,
                    "score": d.score,
                    "source": d.source,
                    "metadata": d.metadata,
                    "chunk_index": d.chunk_index,
                }

    # Add HyDE results
    hyde_docs = expander.expand_and_retrieve(query)
    for d in hyde_docs:
        if d["doc_id"] not in all_docs:
            all_docs[d["doc_id"]] = d

    retrieved = sorted(all_docs.values(), key=lambda x: -x["score"])[:20]
    state["retrieved_docs"] = retrieved
    state["pipeline_trace"] = state.get("pipeline_trace", []) + [
        f"retriever: {len(retrieved)} candidates from {len(state['expanded_queries'])} queries"
    ]
    logger.info("RetrieverAgent: %d docs retrieved.", len(retrieved))
    return state
