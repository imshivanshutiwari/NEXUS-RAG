"""Citation builder: extract [N] markers from answers and map to source documents."""

import re
from typing import Any, Dict, List

from utils.logger import get_logger

logger = get_logger(__name__)


class CitationBuilder:
    """Parse citation markers from generated answers and build citation objects."""

    def extract(
        self, answer: str, documents: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Extract citation numbers from *answer* (e.g., [1], [2])
        and map them to the corresponding *documents*.
        Returns a list of citation dicts.
        """
        citation_numbers = sorted({int(n) for n in re.findall(r"\[(\d+)\]", answer)})
        citations: List[Dict[str, Any]] = []
        for num in citation_numbers:
            idx = num - 1  # [1] → index 0
            if 0 <= idx < len(documents):
                doc = documents[idx]
                citations.append(
                    {
                        "citation_number": num,
                        "doc_id": doc.get("doc_id", ""),
                        "content": doc.get("content", "")[:300],
                        "source": doc.get("source", ""),
                        "chunk_index": doc.get("chunk_index", 0),
                        "url": doc.get("metadata", {}).get("url", ""),
                        "title": doc.get("metadata", {}).get("title", ""),
                    }
                )
        logger.debug("CitationBuilder: %d citations extracted.", len(citations))
        return citations

    def format_bibliography(self, citations: List[Dict[str, Any]]) -> str:
        """Return a formatted bibliography string from a list of citations."""
        if not citations:
            return ""
        lines = ["\n**References:**"]
        for c in citations:
            title = c.get("title") or c.get("source", "Unknown")
            url = c.get("url", "")
            ref = f"[{c['citation_number']}] {title}"
            if url:
                ref += f" — {url}"
            lines.append(ref)
        return "\n".join(lines)
