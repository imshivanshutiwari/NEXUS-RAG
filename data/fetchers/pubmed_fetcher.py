"""PubMed fetcher using NCBI E-utilities (no API key required for basic use)."""

import json
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import requests

from utils.logger import get_logger

logger = get_logger(__name__)

_CACHE_DIR = Path("data/cache/pubmed")
_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"


@dataclass
class Document:
    content: str
    title: str
    url: str
    date: str
    source: str = "pubmed"
    metadata: dict = field(default_factory=dict)


class PubMedFetcher:
    """Fetch real PubMed abstracts via NCBI E-utilities."""

    def __init__(self, cache_dir: Optional[Path] = None) -> None:
        self.cache_dir = cache_dir or _CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "NEXUS-RAG/1.0 (research)"})

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(self, query: str, retmax: int = 200) -> List[str]:
        """Return a list of PubMed IDs matching *query*."""
        cache_path = self.cache_dir / f"search_{self._safe(query)}.json"
        if cache_path.exists():
            return json.loads(cache_path.read_text())

        params = {
            "db": "pubmed",
            "term": query,
            "retmax": retmax,
            "retmode": "json",
        }
        try:
            resp = self.session.get(
                _BASE_URL + "esearch.fcgi", params=params, timeout=15
            )
            resp.raise_for_status()
            pmids = resp.json()["esearchresult"]["idlist"]
            cache_path.write_text(json.dumps(pmids))
            return pmids
        except Exception as exc:
            logger.warning("PubMed search failed for '%s': %s", query, exc)
            return []

    def fetch_abstracts(self, pmids: List[str]) -> List[Document]:
        """Fetch abstracts for a list of PubMed IDs."""
        if not pmids:
            return []
        cache_path = (
            self.cache_dir / f"abstracts_{self._safe(','.join(pmids[:10]))}.json"
        )
        if cache_path.exists():
            return [Document(**d) for d in json.loads(cache_path.read_text())]

        batch_size = 200
        documents: List[Document] = []
        for i in range(0, len(pmids), batch_size):
            batch = pmids[i : i + batch_size]
            params = {
                "db": "pubmed",
                "id": ",".join(batch),
                "rettype": "abstract",
                "retmode": "xml",
            }
            try:
                resp = self.session.post(
                    _BASE_URL + "efetch.fcgi", data=params, timeout=30
                )
                resp.raise_for_status()
                documents.extend(self._parse_xml(resp.text))
                time.sleep(0.4)
            except Exception as exc:
                logger.warning("PubMed fetch_abstracts batch failed: %s", exc)

        cache_path.write_text(json.dumps([d.__dict__ for d in documents], indent=2))
        return documents

    def fetch_full_text(self, pmid: str) -> Optional[Document]:
        """Attempt to fetch full text via PMC Open Access API."""
        cache_path = self.cache_dir / f"fulltext_{pmid}.json"
        if cache_path.exists():
            return Document(**json.loads(cache_path.read_text()))

        # Try PMC OA API
        params = {
            "db": "pmc",
            "term": f"{pmid}[pmid]",
            "retmode": "json",
        }
        try:
            resp = self.session.get(
                _BASE_URL + "esearch.fcgi", params=params, timeout=15
            )
            resp.raise_for_status()
            pmc_ids = resp.json()["esearchresult"]["idlist"]
            if not pmc_ids:
                return None
            pmc_id = pmc_ids[0]
            oa_url = f"https://www.ncbi.nlm.nih.gov/pmc/oai/oai.cgi?verb=GetRecord&identifier=oai:pubmedcentral.nih.gov:{pmc_id}&metadataPrefix=pmc"
            resp2 = self.session.get(oa_url, timeout=30)
            resp2.raise_for_status()
            doc = Document(
                content=resp2.text[:50000],
                title=f"PMC{pmc_id}",
                url=f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmc_id}/",
                date="",
                metadata={"pmid": pmid, "pmc_id": pmc_id},
            )
            cache_path.write_text(json.dumps(doc.__dict__))
            return doc
        except Exception as exc:
            logger.warning("PubMed fetch_full_text failed for %s: %s", pmid, exc)
            return None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _parse_xml(self, xml_text: str) -> List[Document]:
        docs: List[Document] = []
        try:
            root = ET.fromstring(xml_text)
            for article in root.findall(".//PubmedArticle"):
                pmid_el = article.find(".//PMID")
                pmid = pmid_el.text if pmid_el is not None else ""
                title_el = article.find(".//ArticleTitle")
                title = title_el.text or "" if title_el is not None else ""
                abstract_texts = article.findall(".//AbstractText")
                abstract = " ".join((el.text or "") for el in abstract_texts if el.text)
                pub_date = article.find(".//PubDate")
                year = ""
                if pub_date is not None:
                    year_el = pub_date.find("Year")
                    year = year_el.text if year_el is not None else ""
                if abstract:
                    docs.append(
                        Document(
                            content=abstract,
                            title=title,
                            url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                            date=year,
                            metadata={"pmid": pmid},
                        )
                    )
        except ET.ParseError as exc:
            logger.warning("PubMed XML parse error: %s", exc)
        return docs

    @staticmethod
    def _safe(s: str) -> str:
        import re

        return re.sub(r"[^a-z0-9_]", "_", s.lower())[:64]
