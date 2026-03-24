"""Generator agent: build prompt, call Bedrock, extract citations."""

from agents.state import RAGState
from utils.logger import get_logger

logger = get_logger(__name__)


def generator_node(state: RAGState) -> RAGState:
    """
    LangGraph node: Generation.
    Builds context-aware prompt, calls AWS Bedrock Claude-3-Sonnet,
    extracts citations.
    """
    from generation.bedrock_client import BedrockClient
    from generation.prompt_builder import PromptBuilder
    from generation.citation_builder import CitationBuilder

    query = state["query"]
    docs = state.get("reranked_docs", [])

    contexts = [d["content"] for d in docs]
    prompt_builder = PromptBuilder()
    prompt = prompt_builder.build(query, contexts)

    bedrock = BedrockClient()
    try:
        answer = bedrock.generate(prompt, max_tokens=2048, temperature=0.1)
    except Exception as exc:
        logger.error("GeneratorAgent: Bedrock call failed: %s", exc)
        answer = f"[Generation failed: {exc}]"

    citations = CitationBuilder().extract(answer, docs)

    state["generated_answer"] = answer
    state["citations"] = citations
    state["pipeline_trace"] = state.get("pipeline_trace", []) + [
        f"generator: answer={len(answer)} chars, {len(citations)} citations"
    ]
    logger.info("GeneratorAgent: answer generated (%d chars).", len(answer))
    return state
