import time

from app.services.retrieval.retriever import retrieve
from app.services.retrieval.strategy_registry import StrategyRegistry

from app.schemas.evaluation import (
    RetrievedChunk,
    StrategyEvaluationResult,
    RetrievalEvaluationResponse
)


def run_retrieval_evaluation(
    query: str,
    top_k: int
) -> RetrievalEvaluationResponse:

    strategy_results = []

    for strategy in StrategyRegistry.COLLECTIONS:

        try:

            start_time = time.perf_counter()

            results = retrieve(
                query=query,
                strategy=strategy,
                top_k=top_k
            )

            end_time = time.perf_counter()

            retrieval_time_ms = round(
                (end_time - start_time) * 1000,
                2
            )

            retrieved_chunks = []
            scores = []

            for result in results:

                chunk = RetrievedChunk(
                    text=result.payload["text"],
                    score=result.score,
                    source_file=result.payload["source_file"],
                    chunk_id=result.payload["chunk_id"]
                )

                scores.append(
                    chunk.score
                )

                retrieved_chunks.append(
                    chunk
                )

            strategy_result = StrategyEvaluationResult(
                strategy=strategy,
                status="success",

                max_score=round(
                    max(scores),
                    4
                ) if scores else None,

                min_score=round(
                    min(scores),
                    4
                ) if scores else None,

                average_score=round(
                    sum(scores) / len(scores),
                    4
                ) if scores else None,

                retrieval_time_ms=retrieval_time_ms,

                retrieved_chunks=retrieved_chunks
            )

        except ValueError:

            strategy_result = StrategyEvaluationResult(
                strategy=strategy,
                status="collection_not_found",

                max_score=None,
                min_score=None,
                average_score=None,

                retrieval_time_ms=None,

                retrieved_chunks=[]
            )

        strategy_results.append(
            strategy_result
        )

    return RetrievalEvaluationResponse(
        query=query,
        results=strategy_results
    )
