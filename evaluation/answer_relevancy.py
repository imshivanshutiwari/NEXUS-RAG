"""Answer relevancy scorer: measures how well the answer addresses the question."""

from typing import Dict

import numpy as np

from utils.logger import get_logger

logger = get_logger(__name__)


class AnswerRelevancyScorer:
    """
    Score answer relevancy by generating back-questions from the answer
    and comparing their embeddings to the original query embedding.
    """

    def __init__(self) -> None:
        self._embedder = None

    def score(self, query: str, answer: str, n_questions: int = 3) -> Dict[str, float]:
        """
        Generate *n_questions* from the answer and compute average cosine
        similarity with the original *query* embedding.
        Returns {"answer_relevancy": float}.
        """
        embedder = self._get_embedder()
        try:
            back_questions = self._generate_questions(answer, n_questions)
            if not back_questions:
                return {"answer_relevancy": 0.5}

            query_emb = embedder.embed_query(query)
            bq_embs = embedder.embed_documents(back_questions)

            sims = [
                float(embedder.compute_similarity(query_emb, bq_embs[i]))
                for i in range(len(back_questions))
            ]
            relevancy = float(np.mean(sims))
            return {"answer_relevancy": max(0.0, min(1.0, relevancy))}
        except Exception as exc:
            logger.warning("AnswerRelevancyScorer.score failed: %s", exc)
            return {"answer_relevancy": 0.5}

    def _generate_questions(self, answer: str, n: int) -> list:
        """Generate back-questions from the answer using the LLM."""
        try:
            from generation.bedrock_client import BedrockClient

            prompt = (
                f"Generate {n} concise questions that the following text answers. "
                f"Return one question per line, no numbering.\n\nText: {answer[:500]}\n\nQuestions:"
            )
            response = BedrockClient().generate(prompt, max_tokens=256, temperature=0.3)
            return [
                line.strip() for line in response.strip().splitlines() if line.strip()
            ][:n]
        except Exception as exc:
            logger.warning("AnswerRelevancyScorer: question generation failed: %s", exc)
            return []

    def _get_embedder(self):
        if self._embedder is None:
            from ingestion.embedder import DocumentEmbedder

            self._embedder = DocumentEmbedder()
        return self._embedder
