"""ArXiv fetcher using the real arxiv Python library."""

import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

import arxiv

from utils.logger import get_logger

logger = get_logger(__name__)

_CACHE_DIR = Path("data/cache/arxiv")


@dataclass
class Document:
    content: str
    title: str
    url: str
    date: str
    source: str = "arxiv"
    metadata: dict = field(default_factory=dict)


class ArXivFetcher:
    """Fetch real ArXiv papers via the arxiv Python library."""

    _DEFAULT_CATEGORIES = ["cs.AI", "cs.LG", "cs.CL"]

    def __init__(self, cache_dir: Optional[Path] = None) -> None:
        self.cache_dir = cache_dir or _CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.client = arxiv.Client()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_papers(
        self,
        query: str,
        max_results: int = 200,
        categories: Optional[List[str]] = None,
    ) -> List[Document]:
        """Fetch up to *max_results* ArXiv papers matching the query."""
        cats = categories or self._DEFAULT_CATEGORIES
        cat_filter = " OR ".join(f"cat:{c}" for c in cats)
        full_query = f"({query}) AND ({cat_filter})"

        logger.info("ArXiv: query='%s' max=%d", full_query, max_results)
        search = arxiv.Search(
            query=full_query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate,
        )
        documents: List[Document] = []
        for result in self.client.results(search):
            arxiv_id = result.entry_id.split("/")[-1]
            cache_path = self.cache_dir / f"{arxiv_id}.json"
            if cache_path.exists():
                doc = self._load_cache(cache_path)
            else:
                doc = Document(
                    content=result.summary,
                    title=result.title,
                    url=result.entry_id,
                    date=result.published.isoformat() if result.published else "",
                    metadata={
                        "arxiv_id": arxiv_id,
                        "authors": [str(a) for a in result.authors],
                        "categories": result.categories,
                    },
                )
                self._save_cache(cache_path, doc)
            documents.append(doc)
            time.sleep(0.1)
        return documents

    def download_pdf(self, arxiv_id: str) -> bytes:
        """Download the PDF bytes for an ArXiv paper and parse with PyMuPDF."""
        import fitz  # PyMuPDF

        search = arxiv.Search(id_list=[arxiv_id])
        result = next(self.client.results(search))
        pdf_path = result.download_pdf(dirpath=str(self.cache_dir))
        with open(pdf_path, "rb") as fh:
            pdf_bytes = fh.read()
        # Parse text from PDF
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        text = "\n".join(page.get_text() for page in doc)
        logger.info("ArXiv: downloaded PDF for %s (%d chars)", arxiv_id, len(text))
        return pdf_bytes

    def fetch_recent(self, days: int = 30) -> List[Document]:
        """Fetch papers from the last *days* days in default categories."""
        cutoff = (datetime.utcnow() - timedelta(days=days)).strftime("%Y%m%d")
        query = f"submittedDate:[{cutoff}0000 TO *]"
        return self.fetch_papers(query, max_results=100)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _save_cache(self, path: Path, doc: Document) -> None:
        with open(path, "w") as fh:
            json.dump(doc.__dict__, fh, indent=2)

    def _load_cache(self, path: Path) -> Document:
        with open(path) as fh:
            data = json.load(fh)
        return Document(**data)
