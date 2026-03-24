"""LangChain @tool functions used by the NEXUS-RAG agents."""

from typing import Any, Dict, List

from langchain_core.tools import tool

from utils.logger import get_logger

logger = get_logger(__name__)


@tool
def classify_query(query: str) -> str:
    """Classify query into factual | analytical | comparative | conversational."""
    q_lower = query.lower()
    if any(w in q_lower for w in ["compare", "versus", "vs", "difference between"]):
        return "comparative"
    if any(w in q_lower for w in ["analyze", "explain why", "how does", "what causes"]):
        return "analytical"
    if any(w in q_lower for w in ["chat", "tell me", "can you", "help me", "i want"]):
        return "conversational"
    return "factual"


@tool
def detect_language(text: str) -> str:
    """Detect the language of *text* (returns ISO 639-1 code)."""
    try:
        from langdetect import detect  # type: ignore

        return detect(text[:500])
    except Exception:
        return "en"


@tool
def extract_entities(text: str) -> List[str]:
    """Extract named entities from *text* using simple heuristics."""
    import re

    # Capitalised multi-word sequences as entity proxies
    entities = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", text)
    return list(dict.fromkeys(entities))[:20]  # deduplicate, cap at 20


@tool
def hybrid_search(query: str, k: int = 20) -> List[Dict[str, Any]]:
    """Run hybrid BM25 + dense retrieval and return top-k scored documents."""
    from retrieval.hybrid_retriever import HybridRetriever

    retriever = HybridRetriever()
    results = retriever.retrieve(query, k=k)
    return [
        {
            "doc_id": r.doc_id,
            "content": r.content,
            "score": r.score,
            "source": r.source,
            "metadata": r.metadata,
        }
        for r in results
    ]


@tool
def bm25_search(query: str, k: int = 40) -> List[Dict[str, Any]]:
    """BM25 sparse search returning top-k (doc_id, score) pairs."""
    from retrieval.sparse_retriever import SparseRetriever

    retriever = SparseRetriever()
    results = retriever.search(query, k=k)
    return [{"doc_id": r.doc_id, "score": r.score} for r in results]


@tool
def dense_search(query: str, k: int = 20) -> List[Dict[str, Any]]:
    """Dense pgvector ANN search returning top-k documents."""
    from retrieval.dense_retriever import DenseRetriever

    retriever = DenseRetriever()
    results = retriever.search(query, k=k)
    return [
        {"doc_id": r.doc_id, "content": r.content, "score": r.score, "source": r.source}
        for r in results
    ]


@tool
def crossencoder_rerank(
    query: str, documents: List[Dict[str, Any]], top_k: int = 10
) -> List[Dict[str, Any]]:
    """CrossEncoder reranking of candidate documents."""
    from ingestion.indexer import ScoredDocument
    from retrieval.reranker import CrossEncoderReranker

    docs = [
        ScoredDocument(
            doc_id=d["doc_id"], content=d.get("content", ""), score=d.get("score", 0.0)
        )
        for d in documents
    ]
    reranker = CrossEncoderReranker()
    results = reranker.rerank(query, docs, top_k=top_k)
    return [
        {"doc_id": r.doc_id, "content": r.content, "score": r.score} for r in results
    ]


@tool
def colbert_rerank(
    query: str, documents: List[Dict[str, Any]], top_k: int = 5
) -> List[Dict[str, Any]]:
    """ColBERT late interaction reranking."""
    from ingestion.indexer import ScoredDocument
    from retrieval.colbert_reranker import ColBERTReranker

    docs = [
        ScoredDocument(
            doc_id=d["doc_id"], content=d.get("content", ""), score=d.get("score", 0.0)
        )
        for d in documents
    ]
    reranker = ColBERTReranker()
    results = reranker.rerank(query, docs, top_k=top_k)
    return [
        {"doc_id": r.doc_id, "content": r.content, "score": r.score} for r in results
    ]


@tool
def build_prompt(query: str, contexts: List[str]) -> str:
    """Build a RAG prompt from query and context passages."""
    from generation.prompt_builder import PromptBuilder

    return PromptBuilder().build(query, contexts)


@tool
def call_bedrock(prompt: str, max_tokens: int = 2048) -> str:
    """Call AWS Bedrock Claude-3-Sonnet with *prompt*."""
    from generation.bedrock_client import BedrockClient

    return BedrockClient().generate(prompt, max_tokens=max_tokens)


@tool
def extract_citations(
    answer: str, documents: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Extract citation markers [N] from answer and map to source documents."""
    from generation.citation_builder import CitationBuilder

    return CitationBuilder().extract(answer, documents)
