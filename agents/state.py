"""RAGState TypedDict — shared state flowing through the LangGraph pipeline."""

from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict


class Citation(TypedDict):
    doc_id: str
    content: str
    source: str
    chunk_index: int


class RAGASScores(TypedDict):
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


class RAGState(TypedDict):
    query: str
    query_type: str  # factual | analytical | comparative | conversational
    expanded_queries: List[str]
    retrieved_docs: List[Dict[str, Any]]
    reranked_docs: List[Dict[str, Any]]
    generated_answer: str
    citations: List[Citation]
    ragas_scores: Dict[str, float]
    healing_attempts: int
    is_faithful: bool
    final_answer: str
    pipeline_trace: List[str]
