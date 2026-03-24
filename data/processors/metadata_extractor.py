"""Metadata extractor: title, date, source from raw documents."""

import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DocumentMetadata:
    title: str = ""
    date: str = ""
    source: str = ""
    url: str = ""
    authors: list = field(default_factory=list)
    language: str = "en"
    word_count: int = 0
    extra: dict = field(default_factory=dict)


class MetadataExtractor:
    """Extract structured metadata from raw document content."""

    def extract(self, content: str, hint: dict | None = None) -> DocumentMetadata:
        """Extract metadata from *content* with optional hint dict."""
        hint = hint or {}
        title = hint.get("title") or self._extract_title(content)
        date = hint.get("date") or self._extract_date(content)
        source = hint.get("source", "")
        url = hint.get("url", "")
        authors = hint.get("authors", [])
        word_count = len(content.split())
        language = self._detect_language(content)
        return DocumentMetadata(
            title=title,
            date=date,
            source=source,
            url=url,
            authors=authors,
            language=language,
            word_count=word_count,
            extra={
                k: v
                for k, v in hint.items()
                if k not in ("title", "date", "source", "url", "authors")
            },
        )

    def _extract_title(self, content: str) -> str:
        """Use first non-empty line as title heuristic."""
        for line in content.splitlines():
            line = line.strip()
            if len(line) > 10:
                return line[:200]
        return ""

    def _extract_date(self, content: str) -> str:
        """Find first ISO-8601 or common date pattern."""
        patterns = [
            r"\b(\d{4}-\d{2}-\d{2})\b",
            r"\b(\d{2}/\d{2}/\d{4})\b",
            r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}\b",
        ]
        for pattern in patterns:
            m = re.search(pattern, content)
            if m:
                return m.group(0)
        return ""

    def _detect_language(self, content: str) -> str:
        """Heuristic English detection (placeholder for real langdetect)."""
        try:
            from langdetect import detect  # type: ignore

            return detect(content[:500])
        except Exception:
            return "en"
