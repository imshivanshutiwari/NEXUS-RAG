"""Evaluator agent: inline RAGAS scoring (faithfulness, relevancy, precision, recall)."""
from agents.state import RAGState
from utils.logger import get_logger

logger = get_logger(__name__)

_FAITHFULNESS_THRESHOLD = 0.7


def evaluator_node(state: RAGState) -> RAGState:
    """
    LangGraph node: Evaluation.
    Computes RAGAS metrics; sets is_faithful flag.
    If faithfulness < threshold, flags for healing.
    """
    from evaluation.ragas_evaluator import RAGASEvaluator

    query = state["query"]
    answer = state.get("generated_answer", "")
    docs = state.get("reranked_docs", [])
    contexts = [d["content"] for d in docs]

    evaluator = RAGASEvaluator()
    try:
        scores = evaluator.evaluate_response(
            query=query,
            answer=answer,
            contexts=contexts,
            ground_truth=query,  # use query as proxy ground truth
        )
        ragas_scores = {
            "faithfulness": scores.get("faithfulness", 0.0),
            "answer_relevancy": scores.get("answer_relevancy", 0.0),
            "context_precision": scores.get("context_precision", 0.0),
            "context_recall": scores.get("context_recall", 0.0),
        }
    except Exception as exc:
        logger.warning("EvaluatorAgent: RAGAS evaluation failed: %s", exc)
        ragas_scores = {
            "faithfulness": 0.5,
            "answer_relevancy": 0.5,
            "context_precision": 0.5,
            "context_recall": 0.5,
        }

    is_faithful = ragas_scores["faithfulness"] >= _FAITHFULNESS_THRESHOLD
    state["ragas_scores"] = ragas_scores
    state["is_faithful"] = is_faithful
    state["pipeline_trace"] = state.get("pipeline_trace", []) + [
        f"evaluator: faithful={ragas_scores['faithfulness']:.2f} "
        f"relevancy={ragas_scores['answer_relevancy']:.2f}"
    ]
    logger.info(
        "EvaluatorAgent: faithfulness=%.2f is_faithful=%s",
        ragas_scores["faithfulness"],
        is_faithful,
    )
    return state
