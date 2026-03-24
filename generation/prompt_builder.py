"""Dynamic RAG prompt builder."""

from typing import List

from utils.logger import get_logger

logger = get_logger(__name__)

_SYSTEM_PROMPT = (
    "You are NEXUS-RAG, a precise and faithful AI assistant. "
    "Answer the question using ONLY the provided context passages. "
    "Add citation markers [1], [2], ... when referencing specific passages. "
    "If the context does not contain the answer, say so explicitly."
)


class PromptBuilder:
    """Build context-aware prompts for RAG generation."""

    def build(
        self, query: str, contexts: List[str], system: str = _SYSTEM_PROMPT
    ) -> str:
        """Assemble the full prompt with numbered context passages."""
        if not contexts:
            return f"{system}\n\nQuestion: {query}\n\nAnswer:"

        context_block = "\n\n".join(
            f"[{i+1}] {ctx.strip()}" for i, ctx in enumerate(contexts)
        )
        prompt = (
            f"{system}\n\n"
            f"Context Passages:\n{context_block}\n\n"
            f"Question: {query}\n\n"
            f"Answer (cite passages with [N]):"
        )
        logger.debug("PromptBuilder: prompt length=%d chars.", len(prompt))
        return prompt

    def build_healing_prompt(self, query: str, contexts: List[str]) -> str:
        """Fallback prompt used during self-healing."""
        context_block = "\n\n".join(
            f"[{i+1}] {ctx.strip()}" for i, ctx in enumerate(contexts)
        )
        return (
            "You are a helpful assistant. Using the context below, provide the best "
            "possible answer to the question. Be concise and accurate.\n\n"
            f"Context:\n{context_block}\n\nQuestion: {query}\n\nAnswer:"
        )
