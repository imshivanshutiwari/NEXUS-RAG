"""MinHash LSH deduplicator for near-duplicate document removal."""
from typing import Dict, List, Set, Tuple

from utils.logger import get_logger

logger = get_logger(__name__)

_NUM_PERM = 128
_THRESHOLD = 0.85


class Deduplicator:
    """
    Remove near-duplicate documents using MinHash + LSH (datasketch).
    Two documents are considered duplicates if their Jaccard similarity ≥ threshold.
    """

    def __init__(self, threshold: float = _THRESHOLD, num_perm: int = _NUM_PERM) -> None:
        self.threshold = threshold
        self.num_perm = num_perm
        self._lsh = None
        self._seen_keys: Set[str] = set()
        self._init_lsh()

    def _init_lsh(self) -> None:
        try:
            from datasketch import MinHashLSH

            self._lsh = MinHashLSH(threshold=self.threshold, num_perm=self.num_perm)
        except Exception as exc:
            logger.warning("datasketch not available: %s — dedup disabled.", exc)

    def deduplicate(
        self, documents: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        """Return documents with near-duplicates removed. Keeps first occurrence."""
        if self._lsh is None:
            return documents

        from datasketch import MinHash

        unique: List[Dict[str, str]] = []
        for doc in documents:
            key = doc.get("doc_id", doc.get("content", "")[:64])
            if key in self._seen_keys:
                continue
            mh = self._build_minhash(doc.get("content", ""))
            try:
                result = self._lsh.query(mh)
                if result:
                    logger.debug("Deduplicator: duplicate found for doc_id=%s", key)
                    continue
                self._lsh.insert(key, mh)
                self._seen_keys.add(key)
                unique.append(doc)
            except Exception:
                unique.append(doc)

        logger.info(
            "Deduplicator: %d → %d docs (removed %d duplicates).",
            len(documents),
            len(unique),
            len(documents) - len(unique),
        )
        return unique

    def _build_minhash(self, text: str):
        from datasketch import MinHash

        mh = MinHash(num_perm=self.num_perm)
        for word in text.lower().split():
            mh.update(word.encode("utf-8"))
        return mh

    def reset(self) -> None:
        """Clear the LSH index and seen keys."""
        self._seen_keys.clear()
        self._init_lsh()
