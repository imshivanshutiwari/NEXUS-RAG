"""Response validator: hallucination detection via NLI-style checks."""

from typing import Any, Dict, List

from utils.logger import get_logger

logger = get_logger(__name__)


class ResponseValidator:
    """
    Validate generated answers against retrieved context.
    Uses sentence-level NLI (entailment) to detect unsupported claims.
    """

    def __init__(self) -> None:
        self._nli_model = None

    def validate(self, answer: str, contexts: List[str]) -> Dict[str, Any]:
        """
        Check if *answer* is supported by *contexts*.
        Returns a dict with:
          - is_faithful: bool
          - supported_sentences: List[str]
          - unsupported_sentences: List[str]
          - faithfulness_score: float
        """
        sentences = self._split_sentences(answer)
        if not sentences or not contexts:
            return {
                "is_faithful": True,
                "supported_sentences": sentences,
                "unsupported_sentences": [],
                "faithfulness_score": 1.0,
            }

        model = self._get_nli_model()
        context_text = " ".join(contexts)
        supported: List[str] = []
        unsupported: List[str] = []

        for sentence in sentences:
            if not sentence.strip():
                continue
            try:
                result = model(
                    {"text": context_text[:512], "text_pair": sentence},
                    top_k=None,
                )
                entail_score = next(
                    (r["score"] for r in result if r["label"].lower() == "entailment"),
                    0.5,
                )
                if entail_score >= 0.5:
                    supported.append(sentence)
                else:
                    unsupported.append(sentence)
            except Exception:
                supported.append(sentence)

        total = len(supported) + len(unsupported)
        faithfulness_score = len(supported) / total if total > 0 else 1.0
        return {
            "is_faithful": faithfulness_score >= 0.7,
            "supported_sentences": supported,
            "unsupported_sentences": unsupported,
            "faithfulness_score": faithfulness_score,
        }

    def _get_nli_model(self):
        if self._nli_model is None:
            try:
                from transformers import pipeline  # type: ignore

                self._nli_model = pipeline(
                    "text-classification",
                    model="cross-encoder/nli-deberta-v3-small",
                )
                logger.info("ResponseValidator: NLI model loaded.")
            except Exception as exc:
                logger.warning("ResponseValidator: NLI model unavailable: %s", exc)
                self._nli_model = _FallbackNLI()
        return self._nli_model

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        import re

        return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


class _FallbackNLI:
    """Simple keyword-overlap NLI when transformer model unavailable."""

    def __call__(self, inputs: dict, top_k: int = None) -> List[Dict[str, Any]]:
        premise = set(inputs.get("text", "").lower().split())
        hyp = set(inputs.get("text_pair", "").lower().split())
        overlap = len(premise & hyp) / max(len(hyp), 1)
        score = min(overlap * 2, 1.0)
        return [
            {"label": "entailment", "score": score},
            {"label": "contradiction", "score": 1 - score},
        ]
