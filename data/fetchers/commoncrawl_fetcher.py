"""CommonCrawl fetcher via the public Index API and S3 byte-range requests."""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import requests

from utils.logger import get_logger

logger = get_logger(__name__)

_CACHE_DIR = Path("data/cache/commoncrawl")
_INDEX_TEMPLATE = "https://index.commoncrawl.org/{crawl_id}-index"
_S3_BASE = "https://data.commoncrawl.org/"


@dataclass
class Document:
    content: str
    title: str
    url: str
    date: str
    source: str = "commoncrawl"
    metadata: dict = field(default_factory=dict)


class CommonCrawlFetcher:
    """Fetch real web pages from CommonCrawl via its Index API."""

    def __init__(
        self,
        crawl_id: str = "CC-MAIN-2024-10",
        cache_dir: Optional[Path] = None,
    ) -> None:
        self.crawl_id = crawl_id
        self.cache_dir = cache_dir or _CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "NEXUS-RAG/1.0 (research)"})

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search_index(
        self,
        url_pattern: str,
        crawl_id: Optional[str] = None,
    ) -> List[dict]:
        """Query the CommonCrawl index for records matching *url_pattern*."""
        cid = crawl_id or self.crawl_id
        index_url = _INDEX_TEMPLATE.format(crawl_id=cid)
        cache_path = self.cache_dir / f"idx_{self._safe(url_pattern)}.json"
        if cache_path.exists():
            return json.loads(cache_path.read_text())

        params = {"url": url_pattern, "output": "json", "limit": 100}
        records: List[dict] = []
        try:
            resp = self.session.get(index_url, params=params, timeout=30)
            resp.raise_for_status()
            for line in resp.text.strip().splitlines():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
            cache_path.write_text(json.dumps(records))
        except Exception as exc:
            logger.warning(
                "CommonCrawl search_index failed for '%s': %s", url_pattern, exc
            )
        return records

    def fetch_page(self, record_info: dict) -> Optional[Document]:
        """Fetch a single page via byte-range HTTP request to S3."""
        filename = record_info.get("filename", "")
        offset = int(record_info.get("offset", 0))
        length = int(record_info.get("length", 0))
        url = record_info.get("url", "")

        if not filename or not length:
            return None

        cache_path = self.cache_dir / f"page_{self._safe(url)}.json"
        if cache_path.exists():
            return Document(**json.loads(cache_path.read_text()))

        s3_url = _S3_BASE + filename
        headers = {"Range": f"bytes={offset}-{offset + length - 1}"}
        try:
            resp = self.session.get(s3_url, headers=headers, timeout=30)
            resp.raise_for_status()
            raw = resp.content
            # Decompress WARC gzip if needed
            content = self._extract_warc_text(raw)
            doc = Document(
                content=content[:20000],
                title=url,
                url=url,
                date=record_info.get("timestamp", ""),
                metadata={"crawl": self.crawl_id, "offset": offset, "length": length},
            )
            cache_path.write_text(json.dumps(doc.__dict__))
            return doc
        except Exception as exc:
            logger.warning("CommonCrawl fetch_page failed for %s: %s", url, exc)
            return None

    def fetch_domain_sample(self, domain: str, n: int = 100) -> List[Document]:
        """Fetch a sample of *n* pages from a given domain."""
        records = self.search_index(f"*.{domain}/*")
        documents: List[Document] = []
        for record in records[:n]:
            doc = self.fetch_page(record)
            if doc and doc.content.strip():
                documents.append(doc)
        return documents

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_warc_text(self, raw: bytes) -> str:
        """Extract plain text from a WARC/gzip payload."""
        import gzip
        import io

        try:
            with gzip.open(io.BytesIO(raw), "rt", errors="replace") as gz:
                warc_text = gz.read()
        except Exception:
            warc_text = raw.decode("utf-8", errors="replace")

        # Skip WARC headers; keep HTTP body
        parts = warc_text.split("\r\n\r\n", 2)
        body = parts[-1] if len(parts) > 1 else warc_text
        # Strip HTML tags
        body = re.sub(r"<[^>]+>", " ", body)
        body = re.sub(r"\s+", " ", body).strip()
        return body

    @staticmethod
    def _safe(s: str) -> str:
        return re.sub(r"[^a-z0-9_]", "_", s.lower())[:64]
