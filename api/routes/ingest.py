"""Ingest endpoint."""
from fastapi import APIRouter, BackgroundTasks
from api.schemas import IngestRequest, IngestResponse
from utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()

@router.post("/", response_model=IngestResponse)
def trigger_ingest(req: IngestRequest, background_tasks: BackgroundTasks):
    from data.processors.document_store import DocumentStore
    from ingestion.pipeline import IngestionPipeline
    def _run():
        store = DocumentStore()
        store.fetch_all_sources()
        IngestionPipeline().run()
    background_tasks.add_task(_run)
    return IngestResponse(status="accepted", documents_indexed=0, message="Ingestion started in background.")
