"""Healer agent: self-healing via query reformulation and fallback strategies."""
from agents.state import RAGState
from utils.logger import get_logger

logger = get_logger(__name__)

_MAX_ATTEMPTS = 3
_STRATEGIES = ["reformulate", "expand", "switch_mode", "fallback_prompt"]


def healer_node(state: RAGState) -> RAGState:
    """
    LangGraph node: Self-healing.
    Automatically reformulates query or adjusts retrieval strategy when
    faithfulness score is below threshold.
    Max *_MAX_ATTEMPTS* healing iterations.
    """
    attempts = state.get("healing_attempts", 0)
    if attempts >= _MAX_ATTEMPTS:
        logger.warning("HealerAgent: max attempts (%d) reached — finalising.", _MAX_ATTEMPTS)
        state["final_answer"] = state.get("generated_answer", "")
        state["pipeline_trace"] = state.get("pipeline_trace", []) + [
            f"healer: max attempts reached — using best available answer"
        ]
        return state

    strategy = _STRATEGIES[min(attempts, len(_STRATEGIES) - 1)]
    logger.info("HealerAgent: attempt %d strategy=%s", attempts + 1, strategy)

    query = state["query"]

    if strategy == "reformulate":
        new_query = _reformulate(query)
        state["query"] = new_query
    elif strategy == "expand":
        from retrieval.query_expander import QueryExpander
        expanded = QueryExpander().multi_query_expand(query, n=1)
        state["query"] = expanded[0] if expanded else query
    elif strategy == "switch_mode":
        # Switch query type to broaden search
        state["query_type"] = "analytical"
    elif strategy == "fallback_prompt":
        # Use a more lenient generation prompt
        state["query"] = f"Please provide any available information about: {query}"

    state["healing_attempts"] = attempts + 1
    state["retrieved_docs"] = []
    state["reranked_docs"] = []
    state["generated_answer"] = ""
    state["pipeline_trace"] = state.get("pipeline_trace", []) + [
        f"healer: attempt={attempts+1} strategy={strategy}"
    ]
    return state


def _reformulate(query: str) -> str:
    """Use Bedrock to reformulate a query for better retrieval."""
    try:
        from generation.bedrock_client import BedrockClient

        prompt = (
            f"Reformulate the following search query to improve retrieval. "
            f"Return only the reformulated query.\n\nOriginal: {query}\n\nReformulated:"
        )
        return BedrockClient().generate(prompt, max_tokens=128, temperature=0.3).strip()
    except Exception as exc:
        logger.warning("HealerAgent: reformulation failed: %s", exc)
        return query
