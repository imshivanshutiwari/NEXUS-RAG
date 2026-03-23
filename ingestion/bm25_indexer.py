"""BM25 sparse indexer using rank_bm25."""
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from utils.logger import get_logger

logger = get_logger(__name__)

_INDEX_PATH = Path("data/cache/bm25_index.pkl")


class BM25Indexer:
    """Build and query a BM25 sparse index over the document corpus."""

    def __init__(self, index_path: Optional[Path] = None) -> None:
        self.index_path = index_path or _INDEX_PATH
        self._bm25 = None
        self._doc_ids: List[str] = []
        self._corpus: List[List[str]] = []
        if self.index_path.exists():
            self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self, documents: List[Dict[str, str]]) -> None:
        """Build the BM25 index from a list of {doc_id, content} dicts."""
        from rank_bm25 import BM25Okapi

        self._doc_ids = [d["doc_id"] for d in documents]
        self._corpus = [self._tokenize(d["content"]) for d in documents]
        self._bm25 = BM25Okapi(self._corpus)
        self._save()
        logger.info("BM25Indexer: built index over %d documents.", len(documents))

    def search(self, query: str, k: int = 40) -> List[Tuple[str, float]]:
        """Return top-k (doc_id, bm25_score) pairs for *query*."""
        if self._bm25 is None:
            logger.warning("BM25 index not built yet.")
            return []
        tokens = self._tokenize(query)
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(
            zip(self._doc_ids, scores.tolist()), key=lambda x: -x[1]
        )
        return ranked[:k]

    def get_scores_dict(self, query: str) -> Dict[str, float]:
        """Return {doc_id: score} dict for all indexed documents."""
        return dict(self.search(query, k=len(self._doc_ids)))

    def add_documents(self, documents: List[Dict[str, str]]) -> None:
        """Append new documents and rebuild the index."""
        existing = [
            {"doc_id": doc_id, "content": " ".join(tokens)}
            for doc_id, tokens in zip(self._doc_ids, self._corpus)
        ]
        self.build(existing + documents)

    def __len__(self) -> int:
        return len(self._doc_ids)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        import re

        return re.sub(r"[^\w\s]", "", text.lower()).split()

    def _save(self) -> None:
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.index_path, "wb") as fh:
            pickle.dump(
                {
                    "bm25": self._bm25,
                    "doc_ids": self._doc_ids,
                    "corpus": self._corpus,
                },
                fh,
            )

    def _load(self) -> None:
        try:
            with open(self.index_path, "rb") as fh:
                data = pickle.load(fh)
            self._bm25 = data["bm25"]
            self._doc_ids = data["doc_ids"]
            self._corpus = data["corpus"]
            logger.info("BM25Indexer: loaded index with %d docs.", len(self._doc_ids))
        except Exception as exc:
            logger.warning("BM25Indexer: failed to load index: %s", exc)
