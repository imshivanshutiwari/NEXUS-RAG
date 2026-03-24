"""Faithfulness scorer: NLI-based claim-level faithfulness check."""

from typing import Any, Dict, List

from utils.logger import get_logger

logger = get_logger(__name__)


class FaithfulnessScorer:
    """
    Check whether each claim in an answer is entailed by the retrieved context.
    Uses an NLI model (cross-encoder/nli-deberta-v3-small) for entailment scoring.
    """

    def __init__(self) -> None:
        self._pipeline = None

    def score(self, answer: str, contexts: List[str]) -> Dict[str, Any]:
        """
        Score the faithfulness of *answer* given *contexts*.
        Returns:
          - faithfulness_score: float in [0, 1]
          - claims: list of {claim, is_supported, entailment_score}
        """
        claims = self._extract_claims(answer)
        if not claims or not contexts:
            return {"faithfulness_score": 1.0, "claims": []}

        context_text = " ".join(contexts)[:2048]
        nli = self._get_pipeline()
        results = []

        for claim in claims:
            try:
                preds = nli({"text": context_text, "text_pair": claim}, top_k=None)
                entail_score = next(
                    (p["score"] for p in preds if p["label"].lower() == "entailment"),
                    0.5,
                )
            except Exception:
                entail_score = 0.5
            results.append(
                {
                    "claim": claim,
                    "is_supported": entail_score >= 0.5,
                    "entailment_score": entail_score,
                }
            )

        supported = sum(1 for r in results if r["is_supported"])
        score = supported / len(results) if results else 1.0
        return {"faithfulness_score": score, "claims": results}

    def _extract_claims(self, text: str) -> List[str]:
        """Split answer into individual claim sentences."""
        import re

        return [
            s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 10
        ]

    def _get_pipeline(self):
        if self._pipeline is None:
            try:
                from transformers import pipeline  # type: ignore

                self._pipeline = pipeline(
                    "text-classification",
                    model="cross-encoder/nli-deberta-v3-small",
                )
            except Exception as exc:
                logger.warning("FaithfulnessScorer: NLI model unavailable: %s", exc)
                self._pipeline = _KeywordNLI()
        return self._pipeline


class _KeywordNLI:
    def __call__(self, inputs: dict, top_k: int = None) -> List[Dict]:
        premise = set(inputs.get("text", "").lower().split())
        hyp = set(inputs.get("text_pair", "").lower().split())
        overlap = len(premise & hyp) / max(len(hyp), 1)
        score = min(overlap * 2, 1.0)
        return [
            {"label": "entailment", "score": score},
            {"label": "contradiction", "score": 1 - score},
        ]
