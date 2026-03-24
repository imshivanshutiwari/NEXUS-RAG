"""LangGraph StateGraph: 6-node agentic RAG pipeline."""

from typing import Literal

from langgraph.graph import StateGraph, END

from agents.state import RAGState
from agents.router_agent import router_node
from agents.retriever_agent import retriever_node
from agents.reranker_agent import reranker_node
from agents.generator_agent import generator_node
from agents.evaluator_agent import evaluator_node
from agents.healer_agent import healer_node
from utils.logger import get_logger

logger = get_logger(__name__)


def _should_heal(state: RAGState) -> Literal["healer", "end"]:
    """Conditional edge: route to healer if not faithful and attempts remain."""
    if not state.get("is_faithful", True) and state.get("healing_attempts", 0) < 3:
        return "healer"
    # Set final answer before ending
    if not state.get("final_answer"):
        state["final_answer"] = state.get("generated_answer", "")
    return "end"


def _after_healer(state: RAGState) -> Literal["retriever", "end"]:
    """After healing, re-run retrieval unless max attempts exhausted."""
    if state.get("healing_attempts", 0) >= 3:
        return "end"
    return "retriever"


class NEXUSRAGGraph:
    """LangGraph StateGraph with 6 nodes + conditional self-healing edges."""

    def __init__(self) -> None:
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(RAGState)

        # Add 6 nodes
        workflow.add_node("router", router_node)
        workflow.add_node("retriever", retriever_node)
        workflow.add_node("reranker", reranker_node)
        workflow.add_node("generator", generator_node)
        workflow.add_node("evaluator", evaluator_node)
        workflow.add_node("healer", healer_node)

        # Linear edges
        workflow.set_entry_point("router")
        workflow.add_edge("router", "retriever")
        workflow.add_edge("retriever", "reranker")
        workflow.add_edge("reranker", "generator")
        workflow.add_edge("generator", "evaluator")

        # Conditional: evaluator → healer or END
        workflow.add_conditional_edges(
            "evaluator",
            _should_heal,
            {"healer": "healer", "end": END},
        )

        # Conditional: healer → retriever (retry) or END
        workflow.add_conditional_edges(
            "healer",
            _after_healer,
            {"retriever": "retriever", "end": END},
        )

        return workflow.compile()

    def run(self, query: str) -> RAGState:
        """Run the full NEXUS-RAG pipeline for *query*."""
        initial_state: RAGState = {
            "query": query,
            "query_type": "factual",
            "expanded_queries": [],
            "retrieved_docs": [],
            "reranked_docs": [],
            "generated_answer": "",
            "citations": [],
            "ragas_scores": {},
            "healing_attempts": 0,
            "is_faithful": True,
            "final_answer": "",
            "pipeline_trace": [],
        }
        logger.info("NEXUSRAGGraph: running pipeline for query='%s...'", query[:60])
        result = self.graph.invoke(initial_state)
        logger.info(
            "NEXUSRAGGraph: pipeline complete. trace=%s", result.get("pipeline_trace")
        )
        return result
