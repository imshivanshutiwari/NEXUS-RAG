"""CrossEncoder reranker using ms-marco-MiniLM-L-6-v2."""

from typing import List

from ingestion.indexer import ScoredDocument
from utils.config_loader import ConfigLoader
from utils.logger import get_logger

logger = get_logger(__name__)

_DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderReranker:
    """
    Rerank candidate documents using a CrossEncoder model
    (cross-encoder/ms-marco-MiniLM-L-6-v2).
    """

    def __init__(self) -> None:
        try:
            cfg = ConfigLoader("retrieval_config.yaml")
            model_name = cfg.get("reranking.crossencoder_model", _DEFAULT_MODEL)
        except Exception:
            model_name = _DEFAULT_MODEL
        self._model = None
        self._model_name = model_name

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self._model_name)
            logger.info("CrossEncoderReranker: loaded model %s", self._model_name)
        return self._model

    def rerank(
        self,
        query: str,
        documents: List[ScoredDocument],
        top_k: int = 10,
    ) -> List[ScoredDocument]:
        """Rerank *documents* with CrossEncoder and return top_k."""
        if not documents:
            return []
        model = self._get_model()
        pairs = [[query, doc.content] for doc in documents]
        scores = model.predict(pairs)
        ranked = sorted(zip(documents, scores.tolist()), key=lambda x: -x[1])
        results = []
        for doc, score in ranked[:top_k]:
            results.append(
                ScoredDocument(
                    doc_id=doc.doc_id,
                    content=doc.content,
                    score=float(score),
                    source=doc.source,
                    metadata=doc.metadata,
                    chunk_index=doc.chunk_index,
                )
            )
        return results
