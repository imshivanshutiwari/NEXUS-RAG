"""Router agent: classify query type and set retrieval strategy."""
from typing import Any, Dict

from agents.state import RAGState
from utils.logger import get_logger

logger = get_logger(__name__)

_QUERY_TYPES = ("factual", "analytical", "comparative", "conversational")


def router_node(state: RAGState) -> RAGState:
    """
    LangGraph node: Route query.
    Classifies query type, detects language, extracts entities, sets retrieval strategy.
    """
    from agents.tools import classify_query, detect_language, extract_entities

    query = state["query"]
    query_type = classify_query.invoke({"query": query})
    language = detect_language.invoke({"text": query})
    entities = extract_entities.invoke({"text": query})

    state["query_type"] = query_type
    state["pipeline_trace"] = state.get("pipeline_trace", []) + [
        f"router: type={query_type} lang={language} entities={entities[:5]}"
    ]
    logger.info("RouterAgent: type=%s lang=%s", query_type, language)
    return state
