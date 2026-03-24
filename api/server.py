"""FastAPI server for NEXUS-RAG."""

from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes.query import router as query_router
from api.routes.ingest import router as ingest_router
from api.routes.evaluate import router as evaluate_router
from api.routes.monitor import router as monitor_router
from utils.logger import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("NEXUS-RAG API starting up...")
    yield
    logger.info("NEXUS-RAG API shutting down.")


app = FastAPI(
    title="NEXUS-RAG API",
    description="Production-Grade Agentic RAG with LangGraph + AWS Bedrock + pgvector",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(query_router, prefix="/query", tags=["Query"])
app.include_router(ingest_router, prefix="/ingest", tags=["Ingest"])
app.include_router(evaluate_router, prefix="/evaluate", tags=["Evaluate"])
app.include_router(monitor_router, prefix="/monitor", tags=["Monitor"])


@app.get("/health", tags=["Health"])
def health_check():
    return {"status": "online", "service": "NEXUS-RAG"}


def main():
    uvicorn.run("api.server:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()
