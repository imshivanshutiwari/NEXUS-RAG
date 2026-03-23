"""Document embedder: Cohere Embed v3 primary + sentence-transformers fallback."""
import os
from typing import List

import numpy as np
from scipy.spatial.distance import cosine

from utils.logger import get_logger

logger = get_logger(__name__)

_COHERE_MODEL = "embed-english-v3.0"
_LOCAL_MODEL = "multi-qa-mpnet-base-v2"
_EMBED_DIM = 1024


class DocumentEmbedder:
    """
    Embed documents and queries.
    Primary: Cohere Embed v3 (1024-dim).
    Fallback: sentence-transformers multi-qa-mpnet-base-v2 (768-dim, padded to 1024).
    """

    def __init__(self) -> None:
        self._cohere_client = None
        self._local_model = None
        self._use_cohere = self._init_cohere()

    def embed_documents(self, docs: List[str]) -> np.ndarray:
        """Embed a list of document strings. Returns (N, 1024) float32 array."""
        return self.embed_batch(docs, batch_size=96, input_type="search_document")

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string. Returns (1024,) float32 array."""
        result = self._embed_texts([query], input_type="search_query")
        return result[0]

    def embed_batch(
        self,
        docs: List[str],
        batch_size: int = 96,
        input_type: str = "search_document",
    ) -> np.ndarray:
        """Embed in batches with progress tracking. Returns (N, 1024) float32."""
        from tqdm import tqdm

        all_embeddings: List[np.ndarray] = []
        for i in tqdm(range(0, len(docs), batch_size), desc="Embedding"):
            batch = docs[i : i + batch_size]
            all_embeddings.append(self._embed_texts(batch, input_type=input_type))
        if not all_embeddings:
            return np.zeros((0, _EMBED_DIM), dtype=np.float32)
        return np.vstack(all_embeddings).astype(np.float32)

    def compute_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two embedding vectors."""
        return float(1.0 - cosine(a.flatten(), b.flatten()))

    def _init_cohere(self) -> bool:
        api_key = os.getenv("COHERE_API_KEY", "")
        if not api_key:
            logger.warning("COHERE_API_KEY not set — using local fallback embeddings.")
            return False
        try:
            import cohere

            self._cohere_client = cohere.Client(api_key)
            logger.info("Cohere Embed v3 initialised.")
            return True
        except Exception as exc:
            logger.warning("Cohere init failed: %s — using local fallback.", exc)
            return False

    def _get_local_model(self):
        if self._local_model is None:
            from sentence_transformers import SentenceTransformer

            self._local_model = SentenceTransformer(_LOCAL_MODEL)
            logger.info("Loaded local model: %s", _LOCAL_MODEL)
        return self._local_model

    def _embed_texts(self, texts: List[str], input_type: str = "search_document") -> np.ndarray:
        if self._use_cohere and self._cohere_client is not None:
            try:
                response = self._cohere_client.embed(
                    texts=texts,
                    model=_COHERE_MODEL,
                    input_type=input_type,
                )
                return np.array(response.embeddings, dtype=np.float32)
            except Exception as exc:
                logger.warning("Cohere embed failed: %s — falling back to local.", exc)

        model = self._get_local_model()
        embeddings = model.encode(texts, normalize_embeddings=True)
        embeddings = np.array(embeddings, dtype=np.float32)
        if embeddings.shape[1] < _EMBED_DIM:
            pad = np.zeros(
                (embeddings.shape[0], _EMBED_DIM - embeddings.shape[1]), dtype=np.float32
            )
            embeddings = np.concatenate([embeddings, pad], axis=1)
        elif embeddings.shape[1] > _EMBED_DIM:
            embeddings = embeddings[:, :_EMBED_DIM]
        return embeddings
