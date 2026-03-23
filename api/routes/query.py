"""Query endpoint."""
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from api.schemas import QueryRequest, QueryResponse
from utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()

@router.post("/", response_model=QueryResponse)
def run_query(req: QueryRequest):
    from pipeline.main import NEXUSPipeline
    result = NEXUSPipeline().query(req.query)
    return QueryResponse(
        query=result["query"],
        answer=result["answer"],
        citations=result.get("citations", []),
        ragas_scores=result.get("ragas_scores"),
        pipeline_trace=result.get("pipeline_trace", []),
        healing_attempts=result.get("healing_attempts", 0),
        duration_ms=result.get("duration_ms", 0.0),
        query_id=result.get("query_id", ""),
    )

@router.get("/stream")
async def stream_query(q: str):
    from streaming.query_streamer import QueryStreamer
    async def event_gen():
        async for chunk in QueryStreamer().stream_response(q):
            yield chunk
    return StreamingResponse(event_gen(), media_type="text/event-stream")
