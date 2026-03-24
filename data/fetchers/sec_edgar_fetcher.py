"""SEC EDGAR fetcher using real public SEC APIs."""

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import requests

from utils.logger import get_logger

logger = get_logger(__name__)

_CACHE_DIR = Path("data/cache/sec_edgar")
_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
_SEARCH_URL = "https://efts.sec.gov/LATEST/search-index"
_TICKER_URL = "https://www.sec.gov/files/company_tickers.json"


@dataclass
class Document:
    content: str
    title: str
    url: str
    date: str
    source: str = "sec_edgar"
    metadata: dict = field(default_factory=dict)


class SECEDGARFetcher:
    """Fetch real SEC EDGAR filings via the public data API."""

    def __init__(self, cache_dir: Optional[Path] = None) -> None:
        self.cache_dir = cache_dir or _CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "NEXUS-RAG research@nexus-rag.ai"})
        self._ticker_to_cik: dict = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_10k_filings(self, ticker: str, n: int = 5) -> List[Document]:
        """Fetch the last *n* 10-K filings for a company ticker."""
        cik = self._resolve_cik(ticker)
        if not cik:
            logger.warning("SEC: could not resolve CIK for ticker %s", ticker)
            return []
        return self._fetch_filings(ticker, cik, form_type="10-K", n=n)

    def fetch_8k_filings(self, ticker: str) -> List[Document]:
        """Fetch the most recent 8-K filings for a company ticker."""
        cik = self._resolve_cik(ticker)
        if not cik:
            return []
        return self._fetch_filings(ticker, cik, form_type="8-K", n=10)

    def search_filings(self, query: str) -> List[Document]:
        """Full-text search SEC EDGAR for filings matching *query*."""
        cache_path = (
            self.cache_dir / f"search_{re.sub(r'[^a-z0-9]', '_', query.lower())}.json"
        )
        if cache_path.exists():
            return [Document(**d) for d in json.loads(cache_path.read_text())]

        params = {"q": query, "dateRange": "custom", "startdt": "2020-01-01"}
        try:
            resp = self.session.get(_SEARCH_URL, params=params, timeout=20)
            resp.raise_for_status()
            hits = resp.json().get("hits", {}).get("hits", [])
            docs = [self._hit_to_document(h) for h in hits[:20]]
            docs = [d for d in docs if d]
            cache_path.write_text(json.dumps([d.__dict__ for d in docs], indent=2))
            return docs
        except Exception as exc:
            logger.warning("SEC search_filings failed: %s", exc)
            return []

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _resolve_cik(self, ticker: str) -> Optional[str]:
        if ticker in self._ticker_to_cik:
            return self._ticker_to_cik[ticker]
        try:
            resp = self.session.get(_TICKER_URL, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            for entry in data.values():
                self._ticker_to_cik[entry["ticker"]] = str(entry["cik_str"]).zfill(10)
            return self._ticker_to_cik.get(ticker.upper())
        except Exception as exc:
            logger.warning("SEC CIK resolution failed: %s", exc)
            return None

    def _fetch_filings(
        self, ticker: str, cik: str, form_type: str, n: int
    ) -> List[Document]:
        cache_path = self.cache_dir / f"{ticker}_{form_type.replace('-','')}.json"
        if cache_path.exists():
            return [Document(**d) for d in json.loads(cache_path.read_text())]

        url = _SUBMISSIONS_URL.format(cik=cik)
        try:
            resp = self.session.get(url, timeout=20)
            resp.raise_for_status()
            submissions = resp.json()
            filings = submissions.get("filings", {}).get("recent", {})
            forms = filings.get("form", [])
            dates = filings.get("filingDate", [])
            accessions = filings.get("accessionNumber", [])
            primary_docs = filings.get("primaryDocument", [])

            docs: List[Document] = []
            for i, form in enumerate(forms):
                if form != form_type or len(docs) >= n:
                    continue
                accession = accessions[i].replace("-", "")
                primary = primary_docs[i]
                filing_url = (
                    f"https://www.sec.gov/Archives/edgar/data/"
                    f"{int(cik)}/{accession}/{primary}"
                )
                content = self._download_filing_text(filing_url)
                if content:
                    docs.append(
                        Document(
                            content=content[:50000],  # cap per doc
                            title=f"{ticker} {form_type} {dates[i]}",
                            url=filing_url,
                            date=dates[i],
                            metadata={"ticker": ticker, "cik": cik, "form": form_type},
                        )
                    )
                time.sleep(0.5)

            cache_path.write_text(json.dumps([d.__dict__ for d in docs], indent=2))
            return docs
        except Exception as exc:
            logger.warning("SEC _fetch_filings failed for %s: %s", ticker, exc)
            return []

    def _download_filing_text(self, url: str) -> str:
        try:
            resp = self.session.get(url, timeout=30)
            resp.raise_for_status()
            # Strip HTML tags for text filings
            text = re.sub(r"<[^>]+>", " ", resp.text)
            text = re.sub(r"\s+", " ", text).strip()
            return text
        except Exception as exc:
            logger.warning("SEC download_filing_text failed (%s): %s", url, exc)
            return ""

    def _hit_to_document(self, hit: dict) -> Optional[Document]:
        src = hit.get("_source", {})
        content = src.get("file_date", "") + " " + src.get("period_of_report", "")
        return Document(
            content=content or src.get("display_names", [""])[0],
            title=src.get("display_names", ["Unknown"])[0],
            url=src.get("file_path", ""),
            date=src.get("file_date", ""),
            metadata={"form_type": src.get("form_type", "")},
        )
