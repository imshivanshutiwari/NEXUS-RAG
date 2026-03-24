"""Document store: manage document lifecycle (fetch, cache, list, delete)."""

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from utils.logger import get_logger

logger = get_logger(__name__)

_STORE_DIR = Path("data/cache/document_store")


@dataclass
class Document:
    content: str
    title: str
    url: str
    date: str
    source: str = "unknown"
    doc_id: str = ""
    metadata: dict = field(default_factory=dict)


class DocumentStore:
    """Persist and retrieve raw documents across fetching sessions."""

    def __init__(self, store_dir: Optional[Path] = None) -> None:
        self.store_dir = store_dir or _STORE_DIR
        self.store_dir.mkdir(parents=True, exist_ok=True)

    def save(self, doc: Document) -> str:
        """Save a document and return its doc_id."""
        import hashlib

        if not doc.doc_id:
            doc.doc_id = hashlib.sha256(doc.content.encode()).hexdigest()[:16]
        path = self.store_dir / f"{doc.source}_{doc.doc_id}.json"
        with open(path, "w") as fh:
            json.dump(doc.__dict__, fh, indent=2)
        return doc.doc_id

    def load(self, doc_id: str) -> Optional[Document]:
        """Load a document by doc_id."""
        for path in self.store_dir.glob(f"*_{doc_id}.json"):
            with open(path) as fh:
                return Document(**json.load(fh))
        return None

    def list_documents(self, source: Optional[str] = None) -> List[Document]:
        """List all stored documents, optionally filtered by source."""
        docs: List[Document] = []
        for path in sorted(self.store_dir.glob("*.json")):
            try:
                with open(path) as fh:
                    data = json.load(fh)
                doc = Document(**data)
                if source is None or doc.source == source:
                    docs.append(doc)
            except Exception as exc:
                logger.warning(
                    "DocumentStore.list_documents: bad file %s: %s", path, exc
                )
        return docs

    def delete(self, doc_id: str) -> bool:
        """Delete a document by doc_id. Returns True if deleted."""
        for path in self.store_dir.glob(f"*_{doc_id}.json"):
            path.unlink()
            return True
        return False

    def count(self, source: Optional[str] = None) -> int:
        return len(self.list_documents(source))

    def fetch_all_sources(self) -> None:
        """Trigger all real data source fetchers and persist documents."""
        from data.fetchers.wikipedia_fetcher import WikipediaFetcher
        from data.fetchers.arxiv_fetcher import ArXivFetcher
        from data.fetchers.sec_edgar_fetcher import SECEDGARFetcher
        from data.fetchers.pubmed_fetcher import PubMedFetcher
        from data.fetchers.commoncrawl_fetcher import CommonCrawlFetcher
        from utils.config_loader import ConfigLoader

        cfg = ConfigLoader("pipeline_config.yaml")

        logger.info("Fetching Wikipedia articles...")
        wiki_docs = WikipediaFetcher().fetch_articles(
            cfg.get("corpus.wikipedia_topics", ["AI", "machine_learning"]),
            n_per_topic=cfg.get("corpus.n_docs_per_source", 50),
        )
        for doc in wiki_docs:
            self.save(
                Document(
                    content=doc.content,
                    title=doc.title,
                    url=doc.url,
                    date=doc.date,
                    source="wikipedia",
                    metadata=doc.metadata,
                )
            )
        logger.info("Saved %d Wikipedia documents.", len(wiki_docs))

        logger.info("Fetching ArXiv papers...")
        arxiv_fetcher = ArXivFetcher()
        for query in cfg.get("corpus.arxiv_queries", ["RAG"]):
            papers = arxiv_fetcher.fetch_papers(query, max_results=50)
            for doc in papers:
                self.save(
                    Document(
                        content=doc.content,
                        title=doc.title,
                        url=doc.url,
                        date=doc.date,
                        source="arxiv",
                        metadata=doc.metadata,
                    )
                )
        logger.info("ArXiv papers fetched.")

        logger.info("Fetching SEC EDGAR filings...")
        sec = SECEDGARFetcher()
        for ticker in cfg.get("corpus.sec_tickers", ["MSFT"]):
            for doc in sec.fetch_10k_filings(ticker, n=2):
                self.save(
                    Document(
                        content=doc.content,
                        title=doc.title,
                        url=doc.url,
                        date=doc.date,
                        source="sec_edgar",
                        metadata=doc.metadata,
                    )
                )
        logger.info("SEC filings fetched.")

        logger.info("Fetching PubMed abstracts...")
        pubmed = PubMedFetcher()
        for query in cfg.get("corpus.pubmed_queries", ["NLP clinical"]):
            pmids = pubmed.search(query, retmax=50)
            for doc in pubmed.fetch_abstracts(pmids):
                self.save(
                    Document(
                        content=doc.content,
                        title=doc.title,
                        url=doc.url,
                        date=doc.date,
                        source="pubmed",
                        metadata=doc.metadata,
                    )
                )
        logger.info("PubMed abstracts fetched.")

        logger.info("Fetching CommonCrawl pages...")
        cc = CommonCrawlFetcher()
        for domain in ["en.wikipedia.org", "arxiv.org"]:
            for doc in cc.fetch_domain_sample(domain, n=20):
                self.save(
                    Document(
                        content=doc.content,
                        title=doc.title,
                        url=doc.url,
                        date=doc.date,
                        source="commoncrawl",
                        metadata=doc.metadata,
                    )
                )
        logger.info("CommonCrawl pages fetched.")
        logger.info("Total documents in store: %d", self.count())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NEXUS-RAG document store manager")
    parser.add_argument("--fetch", action="store_true", help="Fetch all data sources")
    parser.add_argument("--count", action="store_true", help="Print document count")
    args = parser.parse_args()

    store = DocumentStore()
    if args.fetch:
        store.fetch_all_sources()
    elif args.count:
        print(f"Total documents: {store.count()}")
    else:
        parser.print_help()
