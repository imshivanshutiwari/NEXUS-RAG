"""PDF processor using PyMuPDF with layout analysis."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Document:
    content: str
    title: str
    url: str
    date: str
    source: str = "pdf"
    metadata: dict = field(default_factory=dict)


class PDFProcessor:
    """Parse PDFs using PyMuPDF with basic layout analysis."""

    def process_file(self, path: Path) -> Optional[Document]:
        """Extract text from a PDF file preserving page structure."""
        try:
            import fitz  # PyMuPDF

            doc = fitz.open(str(path))
            pages_text: List[str] = []
            for page_num, page in enumerate(doc, start=1):
                blocks = page.get_text("blocks")
                # Sort blocks top-to-bottom, left-to-right
                blocks.sort(key=lambda b: (round(b[1] / 10), b[0]))
                page_text = "\n".join(b[4].strip() for b in blocks if b[4].strip())
                pages_text.append(f"[PAGE {page_num}]\n{page_text}")
            full_text = "\n\n".join(pages_text)
            title = path.stem.replace("_", " ")
            return Document(
                content=full_text,
                title=title,
                url=str(path),
                date="",
                metadata={"num_pages": len(doc), "file_size": path.stat().st_size},
            )
        except Exception as exc:
            logger.warning("PDFProcessor.process_file failed for %s: %s", path, exc)
            return None

    def process_bytes(self, pdf_bytes: bytes, title: str = "") -> Optional[Document]:
        """Extract text from raw PDF bytes."""
        try:
            import fitz

            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            text = "\n\n".join(
                f"[PAGE {i+1}]\n{page.get_text()}" for i, page in enumerate(doc)
            )
            return Document(
                content=text,
                title=title,
                url="",
                date="",
                metadata={"num_pages": len(doc)},
            )
        except Exception as exc:
            logger.warning("PDFProcessor.process_bytes failed: %s", exc)
            return None
