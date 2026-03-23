"""Pydantic request/response schemas for the NEXUS-RAG API."""
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000, description="User query")
    k: int = Field(default=5, ge=1, le=20, description="Number of context documents")
    stream: bool = Field(default=False, description="Stream response via SSE")


class Citation(BaseModel):
    citation_number: int
    doc_id: str
    content: str
    source: str
    chunk_index: int
    url: str = ""
    title: str = ""


class RAGASScores(BaseModel):
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


class QueryResponse(BaseModel):
    query: str
    answer: str
    citations: List[Citation] = []
    ragas_scores: Optional[RAGASScores] = None
    pipeline_trace: List[str] = []
    healing_attempts: int = 0
    duration_ms: float = 0.0
    query_id: str = ""


class IngestRequest(BaseModel):
    sources: Optional[List[str]] = Field(
        default=None,
        description="Sources to fetch: wikipedia|arxiv|sec_edgar|pubmed|commoncrawl (all if None)",
    )


class IngestResponse(BaseModel):
    status: str
    documents_indexed: int
    message: str = ""


class EvaluateRequest(BaseModel):
    query: str
    answer: str
    contexts: List[str]
    ground_truth: Optional[str] = None


class EvaluateResponse(BaseModel):
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


class MonitorResponse(BaseModel):
    drift_score: float
    drift_status: str
    rolling_faithfulness: float
    rolling_relevancy: float
    latency_stats: Dict[str, Any] = {}
    recent_alerts: List[Dict[str, Any]] = []
