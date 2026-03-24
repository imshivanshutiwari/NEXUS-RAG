"""Server-Sent Events (SSE) streaming for RAG query responses."""

import asyncio
import json
from typing import AsyncIterator

from utils.logger import get_logger

logger = get_logger(__name__)


class QueryStreamer:
    """Stream RAG pipeline responses token-by-token via SSE."""

    async def stream_response(self, query: str) -> AsyncIterator[str]:
        """
        Run the NEXUS-RAG pipeline and yield SSE-formatted events.
        Yields: data: {"type": "token"|"metadata"|"done", ...}\n\n
        """
        try:
            from generation.bedrock_client import BedrockClient
            from generation.prompt_builder import PromptBuilder
            from retrieval.hybrid_retriever import HybridRetriever
            from retrieval.reranker import CrossEncoderReranker

            # Retrieve
            retriever = HybridRetriever()
            docs = retriever.retrieve(query, k=10)
            reranker = CrossEncoderReranker()
            docs = reranker.rerank(query, docs, top_k=5)

            contexts = [d.content for d in docs]
            prompt = PromptBuilder().build(query, contexts)
            bedrock = BedrockClient()

            # Stream tokens
            for token in bedrock.stream_generate(prompt):
                event = json.dumps({"type": "token", "text": token})
                yield f"data: {event}\n\n"
                await asyncio.sleep(0)

            # Final metadata event
            meta = json.dumps(
                {
                    "type": "metadata",
                    "n_docs": len(docs),
                    "sources": list({d.source for d in docs}),
                }
            )
            yield f"data: {meta}\n\n"
            yield 'data: {"type": "done"}\n\n'

        except Exception as exc:
            error = json.dumps({"type": "error", "message": str(exc)})
            yield f"data: {error}\n\n"
            logger.error("QueryStreamer: error streaming response: %s", exc)
