"""Query expander: HyDE + multi-query expansion."""
from typing import List

from utils.logger import get_logger

logger = get_logger(__name__)


class QueryExpander:
    """
    Expand queries using:
    - HyDE (Hypothetical Document Embeddings): generate a hypothetical answer and embed it
    - Multi-query expansion: generate N paraphrases of the query
    """

    def __init__(self) -> None:
        from generation.bedrock_client import BedrockClient

        self.llm = BedrockClient()

    def hyde_expand(self, query: str) -> str:
        """Generate a hypothetical document that answers the query."""
        prompt = (
            f"Write a concise, factual passage (2-4 sentences) that directly answers "
            f"the following question. Do not add disclaimers.\n\nQuestion: {query}\n\nPassage:"
        )
        try:
            return self.llm.generate(prompt, max_tokens=256, temperature=0.3)
        except Exception as exc:
            logger.warning("HyDE generation failed: %s", exc)
            return query

    def multi_query_expand(self, query: str, n: int = 3) -> List[str]:
        """Generate *n* different reformulations of the query."""
        prompt = (
            f"Generate {n} different but semantically equivalent versions of the "
            f"following search query. Return one per line, no numbering.\n\nQuery: {query}\n\nVersions:"
        )
        try:
            response = self.llm.generate(prompt, max_tokens=256, temperature=0.5)
            lines = [line.strip() for line in response.strip().splitlines() if line.strip()]
            return lines[:n]
        except Exception as exc:
            logger.warning("Multi-query expansion failed: %s", exc)
            return [query]

    def expand_and_retrieve(self, query: str) -> List[dict]:
        """
        Run HyDE + multi-query expansion then collect all hybrid retrieval results.
        Returns deduplicated, ranked list of documents.
        """
        from retrieval.hybrid_retriever import HybridRetriever
        from ingestion.embedder import DocumentEmbedder

        retriever = HybridRetriever()
        embedder = DocumentEmbedder()

        expanded = [query] + self.multi_query_expand(query, n=3)
        hyde_doc = self.hyde_expand(query)

        all_results: dict = {}

        # Retrieve for each expanded query
        for q in expanded:
            for doc in retriever.retrieve(q, k=10):
                if doc.doc_id not in all_results:
                    all_results[doc.doc_id] = doc

        # Retrieve for HyDE hypothetical doc
        if hyde_doc != query:
            hyde_vec = embedder.embed_query(hyde_doc)
            for doc in retriever.dense_retriever.search_by_vector(hyde_vec, k=10):
                if doc.doc_id not in all_results:
                    all_results[doc.doc_id] = doc

        # Rank by score
        ranked = sorted(all_results.values(), key=lambda d: -d.score)
        logger.info("QueryExpander: %d unique docs from expanded retrieval.", len(ranked))
        return [
            {
                "doc_id": d.doc_id,
                "content": d.content,
                "score": d.score,
                "source": d.source,
                "metadata": d.metadata,
            }
            for d in ranked
        ]
