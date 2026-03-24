"""End-to-end pipeline entry: imports and uses all modules."""

import time
from typing import Any, Dict

from agents.graph import NEXUSRAGGraph
from data.processors.document_store import DocumentStore
from evaluation.ragas_evaluator import RAGASEvaluator
from generation.bedrock_client import BedrockClient
from generation.prompt_builder import PromptBuilder
from ingestion.embedder import DocumentEmbedder
from ingestion.indexer import PGVectorIndexer
from ingestion.bm25_indexer import BM25Indexer
from monitoring.drift_detector import EmbeddingDriftDetector
from monitoring.latency_tracker import get_tracker
from monitoring.quality_monitor import QualityMonitor
from monitoring.alert_manager import AlertManager
from monitoring.self_healer import SelfHealer
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.colbert_reranker import ColBERTReranker
from streaming.pipeline_events import get_events
from utils.logger import get_logger

logger = get_logger(__name__)


class NEXUSPipeline:
    """
    Top-level orchestrator that wires together all NEXUS-RAG modules.
    This is the single entry point for end-to-end query processing.
    """

    def __init__(self) -> None:
        self.graph = NEXUSRAGGraph()
        self.quality_monitor = QualityMonitor()
        self.alert_manager = AlertManager()
        self.drift_detector = EmbeddingDriftDetector()
        self.latency_tracker = get_tracker()

        # Register self-healer for alerts
        healer = SelfHealer()
        self.alert_manager.register_handler(healer.trigger_on_alert)

        logger.info("NEXUSPipeline: all modules initialised.")

    def query(self, question: str) -> Dict[str, Any]:
        """Process a single question end-to-end and return the full result dict."""
        query_id = str(int(time.time() * 1000))
        self.latency_tracker.start("pipeline", query_id)

        state = self.graph.run(question)

        duration_ms = self.latency_tracker.stop("pipeline", query_id)

        # Record quality
        scores = state.get("ragas_scores", {})
        if scores:
            self.quality_monitor.record(scores)
            self.alert_manager.check_faithfulness(scores.get("faithfulness", 1.0))
        self.alert_manager.check_latency(duration_ms)

        return {
            "query": question,
            "answer": state.get("final_answer", ""),
            "citations": state.get("citations", []),
            "ragas_scores": scores,
            "pipeline_trace": state.get("pipeline_trace", []),
            "healing_attempts": state.get("healing_attempts", 0),
            "duration_ms": duration_ms,
            "query_id": query_id,
        }


def main() -> None:
    """CLI entry point: run a sample query."""
    pipeline = NEXUSPipeline()
    result = pipeline.query("What is Retrieval-Augmented Generation?")
    logger.info("Pipeline result: %s", result.get("answer", "")[:200])


if __name__ == "__main__":
    main()
