"""Full ingestion pipeline: fetch → chunk → embed → index."""

from pathlib import Path
from typing import List

from utils.logger import get_logger

logger = get_logger(__name__)


class IngestionPipeline:
    """
    Orchestrates the full ingestion flow:
    DocumentStore → TextChunker → Deduplicator → DocumentEmbedder → PGVectorIndexer + BM25Indexer
    """

    def __init__(self) -> None:
        from data.processors.document_store import DocumentStore
        from data.processors.text_chunker import TextChunker
        from ingestion.embedder import DocumentEmbedder
        from ingestion.indexer import PGVectorIndexer
        from ingestion.bm25_indexer import BM25Indexer
        from ingestion.deduplicator import Deduplicator

        self.store = DocumentStore()
        self.chunker = TextChunker()
        self.embedder = DocumentEmbedder()
        self.pg_indexer = PGVectorIndexer()
        self.bm25_indexer = BM25Indexer()
        self.deduplicator = Deduplicator()

    def run(self) -> None:
        """Execute the full ingestion pipeline."""
        logger.info("IngestionPipeline: starting.")

        # Ensure schema is ready
        if self.pg_indexer._conn is not None:
            self.pg_indexer.create_schema()

        # Load raw documents
        raw_docs = self.store.list_documents()
        logger.info("IngestionPipeline: loaded %d raw documents.", len(raw_docs))
        if not raw_docs:
            logger.warning("No documents found. Run 'make fetch' first.")
            return

        # Chunk documents
        chunks: List[dict] = []
        for doc in raw_docs:
            for chunk in self.chunker.chunk(
                doc.content,
                doc_id=doc.doc_id,
                metadata={"source": doc.source, "title": doc.title, "url": doc.url},
            ):
                chunks.append(
                    {
                        "content": chunk.content,
                        "doc_id": f"{doc.doc_id}_{chunk.chunk_index}",
                        "source": doc.source,
                        "chunk_index": chunk.chunk_index,
                        "metadata": chunk.metadata,
                    }
                )
        logger.info("IngestionPipeline: %d chunks created.", len(chunks))

        # Deduplicate
        chunks = self.deduplicator.deduplicate(chunks)
        logger.info("IngestionPipeline: %d chunks after deduplication.", len(chunks))

        # Embed
        texts = [c["content"] for c in chunks]
        embeddings = self.embedder.embed_documents(texts)
        logger.info("IngestionPipeline: embeddings shape %s.", embeddings.shape)

        # Index in pgvector
        if self.pg_indexer._conn is not None:
            self.pg_indexer.insert_documents(chunks, embeddings)
        else:
            logger.warning("PostgreSQL unavailable — skipping pgvector indexing.")

        # Build BM25 index
        self.bm25_indexer.build(chunks)
        logger.info("IngestionPipeline: complete. %d chunks indexed.", len(chunks))


def main() -> None:
    pipeline = IngestionPipeline()
    pipeline.run()


if __name__ == "__main__":
    main()
