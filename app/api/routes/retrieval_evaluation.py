from fastapi import APIRouter

from app.services.evaluation.retrieval_evaluator import (
    run_retrieval_evaluation
)

from app.schemas.evaluation import (
    RetrievalEvaluationRequest,
    RetrievalEvaluationResponse
)

eval_router = APIRouter()


@eval_router.post(
    "/evaluate/retrieval",
    response_model=RetrievalEvaluationResponse
)
def evaluate_retrieval(
    request: RetrievalEvaluationRequest
):

    return run_retrieval_evaluation(
        query=request.query,
        top_k=request.top_k,
        reranking_enabled=request.reranking_enabled,
        reranker_model=request.reranker_model,
        candidate_count=request.candidate_count
    )
