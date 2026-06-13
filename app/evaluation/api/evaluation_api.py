from fastapi import APIRouter

from app.evaluation.retriever.retrieval_evaluator import (
    run_retrieval_evaluation
)

from app.evaluation.models.evaluation_models import (
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
        top_k=request.top_k
    )