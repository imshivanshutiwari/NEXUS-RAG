"""Evaluate endpoint."""
from fastapi import APIRouter
from api.schemas import EvaluateRequest, EvaluateResponse
from evaluation.ragas_evaluator import RAGASEvaluator

router = APIRouter()

@router.post("/", response_model=EvaluateResponse)
def evaluate(req: EvaluateRequest):
    scores = RAGASEvaluator().evaluate_response(
        query=req.query, answer=req.answer, contexts=req.contexts, ground_truth=req.ground_truth
    )
    return EvaluateResponse(**scores)
