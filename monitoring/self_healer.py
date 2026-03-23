"""Self-healer: auto-retrain and re-index trigger when quality degrades."""
from utils.logger import get_logger

logger = get_logger(__name__)


class SelfHealer:
    """
    Responds to quality or drift alerts by triggering re-ingestion
    and re-indexing of the corpus.
    """

    def reindex(self) -> None:
        """Trigger a full re-ingestion and re-indexing pipeline."""
        logger.info("SelfHealer: triggering re-index pipeline...")
        try:
            from ingestion.pipeline import IngestionPipeline

            pipeline = IngestionPipeline()
            pipeline.run()
            logger.info("SelfHealer: re-indexing complete.")
        except Exception as exc:
            logger.error("SelfHealer.reindex failed: %s", exc)

    def refresh_bm25(self) -> None:
        """Rebuild the BM25 index from the current document store."""
        logger.info("SelfHealer: refreshing BM25 index...")
        try:
            from data.processors.document_store import DocumentStore
            from ingestion.bm25_indexer import BM25Indexer

            store = DocumentStore()
            docs = store.list_documents()
            indexer = BM25Indexer()
            indexer.build(
                [{"doc_id": d.doc_id, "content": d.content} for d in docs]
            )
            logger.info("SelfHealer: BM25 index refreshed with %d docs.", len(docs))
        except Exception as exc:
            logger.error("SelfHealer.refresh_bm25 failed: %s", exc)

    def trigger_on_alert(self, alert) -> None:
        """Callback for AlertManager — trigger appropriate healing action."""
        if alert.name == "drift_detected":
            self.reindex()
        elif alert.name == "faithfulness_low":
            logger.warning("SelfHealer: low faithfulness alert — consider corpus refresh.")
        elif alert.name == "latency_high":
            logger.warning("SelfHealer: high latency — consider HNSW ef_search tuning.")
