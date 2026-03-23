"""Wikipedia fetcher using the real Wikipedia API."""
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import requests

from utils.logger import get_logger

logger = get_logger(__name__)

_CACHE_DIR = Path("data/cache/wikipedia")
_API_URL = "https://en.wikipedia.org/w/api.php"


@dataclass
class Document:
    content: str
    title: str
    url: str
    date: str
    source: str = "wikipedia"
    metadata: dict = field(default_factory=dict)


class WikipediaFetcher:
    """Fetch real Wikipedia articles via the MediaWiki API."""

    def __init__(self, cache_dir: Optional[Path] = None) -> None:
        self.cache_dir = cache_dir or _CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "NEXUS-RAG/1.0 (research project)"})

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_articles(self, topics: List[str], n_per_topic: int = 50) -> List[Document]:
        """Fetch up to *n_per_topic* Wikipedia articles per topic."""
        documents: List[Document] = []
        for topic in topics:
            logger.info("Wikipedia: fetching topic='%s' n=%d", topic, n_per_topic)
            page_ids = self._search_page_ids(topic, n_per_topic)
            for pid in page_ids:
                doc = self.fetch_full_article(str(pid))
                if doc and doc.content:
                    documents.append(doc)
            time.sleep(0.3)  # polite rate limiting
        return documents

    def fetch_full_article(self, page_id: str) -> Optional[Document]:
        """Fetch the full plain-text extract of a Wikipedia article by page ID."""
        cache_path = self.cache_dir / f"{page_id}.json"
        if cache_path.exists():
            return self._load_cache(cache_path)

        params = {
            "action": "query",
            "pageids": page_id,
            "prop": "extracts|info",
            "explaintext": True,
            "inprop": "url",
            "format": "json",
        }
        try:
            resp = self.session.get(_API_URL, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            pages = data.get("query", {}).get("pages", {})
            page = next(iter(pages.values()))
            doc = Document(
                content=page.get("extract", ""),
                title=page.get("title", ""),
                url=page.get("fullurl", f"https://en.wikipedia.org/?curid={page_id}"),
                date=page.get("touched", ""),
                metadata={"page_id": page_id},
            )
            self._save_cache(cache_path, doc)
            return doc
        except Exception as exc:
            logger.warning("Wikipedia fetch_full_article failed for %s: %s", page_id, exc)
            return None

    def fetch_category(self, category: str, depth: int = 2) -> List[Document]:
        """Recursively fetch articles in a Wikipedia category up to *depth* levels."""
        page_ids = self._category_members(category, depth)
        return [
            doc
            for pid in page_ids
            if (doc := self.fetch_full_article(str(pid))) and doc.content
        ]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _search_page_ids(self, query: str, n: int) -> List[int]:
        params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": min(n, 50),
            "format": "json",
        }
        try:
            resp = self.session.get(_API_URL, params=params, timeout=15)
            resp.raise_for_status()
            return [r["pageid"] for r in resp.json()["query"]["search"]]
        except Exception as exc:
            logger.warning("Wikipedia search failed for '%s': %s", query, exc)
            return []

    def _category_members(self, category: str, depth: int) -> List[int]:
        if depth == 0:
            return []
        params = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": f"Category:{category}",
            "cmlimit": 50,
            "cmtype": "page",
            "format": "json",
        }
        try:
            resp = self.session.get(_API_URL, params=params, timeout=15)
            resp.raise_for_status()
            members = resp.json()["query"]["categorymembers"]
            ids = [m["pageid"] for m in members]
            if depth > 1:
                sub_params = dict(params, cmtype="subcat")
                sub_resp = self.session.get(_API_URL, params=sub_params, timeout=15)
                sub_resp.raise_for_status()
                for sub in sub_resp.json()["query"]["categorymembers"]:
                    sub_title = sub["title"].replace("Category:", "")
                    ids.extend(self._category_members(sub_title, depth - 1))
            return ids
        except Exception as exc:
            logger.warning("Wikipedia category members failed: %s", exc)
            return []

    def _save_cache(self, path: Path, doc: Document) -> None:
        with open(path, "w") as fh:
            json.dump(doc.__dict__, fh, indent=2)

    def _load_cache(self, path: Path) -> Document:
        with open(path) as fh:
            data = json.load(fh)
        return Document(**data)

    def cache_locally(self, path: str = "data/cache/wikipedia/") -> None:
        """Set the local cache directory."""
        self.cache_dir = Path(path)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
