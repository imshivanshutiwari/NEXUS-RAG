"""Semantic text chunker: splits documents into overlapping semantic chunks."""
from dataclasses import dataclass, field
from typing import List

import re

from utils.config_loader import ConfigLoader
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class TextChunk:
    content: str
    chunk_index: int
    doc_id: str
    metadata: dict = field(default_factory=dict)


class TextChunker:
    """
    Semantic chunker that splits text into chunks near *chunk_size* tokens,
    respecting sentence boundaries with *chunk_overlap* token overlap.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        strategy: str = "semantic",
    ) -> None:
        try:
            cfg = ConfigLoader("pipeline_config.yaml")
            self.chunk_size = cfg.get("chunking.chunk_size", chunk_size)
            self.chunk_overlap = cfg.get("chunking.chunk_overlap", chunk_overlap)
            self.strategy = cfg.get("chunking.strategy", strategy)
        except Exception:
            self.chunk_size = chunk_size
            self.chunk_overlap = chunk_overlap
            self.strategy = strategy

    def chunk(self, text: str, doc_id: str = "", metadata: dict | None = None) -> List[TextChunk]:
        """Split *text* into overlapping semantic chunks."""
        if not text.strip():
            return []

        sentences = self._split_sentences(text)
        chunks: List[TextChunk] = []
        current_tokens: List[str] = []
        chunk_idx = 0

        for sentence in sentences:
            tokens = sentence.split()
            if len(current_tokens) + len(tokens) > self.chunk_size and current_tokens:
                chunk_text = " ".join(current_tokens)
                chunks.append(
                    TextChunk(
                        content=chunk_text,
                        chunk_index=chunk_idx,
                        doc_id=doc_id,
                        metadata=metadata or {},
                    )
                )
                chunk_idx += 1
                # Overlap: keep last chunk_overlap tokens
                current_tokens = current_tokens[-self.chunk_overlap :]
            current_tokens.extend(tokens)

        if current_tokens:
            chunks.append(
                TextChunk(
                    content=" ".join(current_tokens),
                    chunk_index=chunk_idx,
                    doc_id=doc_id,
                    metadata=metadata or {},
                )
            )

        logger.debug("TextChunker: %d chunks from doc_id=%s", len(chunks), doc_id)
        return chunks

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences using regex."""
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        return [s.strip() for s in sentences if s.strip()]
