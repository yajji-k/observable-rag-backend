import time

from openinference.semconv.trace import OpenInferenceSpanKindValues

from app.observability.tracing import (
    set_attributes,
    set_input,
    set_output,
    span_kind,
    tracer,
)
from app.services.retrieval.retriever import retrieve
from app.services.retrieval.strategy_registry import StrategyRegistry

from app.schemas.evaluation import (
    RetrievedChunk,
    StrategyEvaluationResult,
    RetrievalEvaluationResponse
)


def run_retrieval_evaluation(
    query: str,
    top_k: int,
    benchmark_run_id: str | None = None,
    benchmark_query_id: str | None = None
) -> RetrievalEvaluationResponse:
    with tracer.start_as_current_span(
        "retrieval.evaluate",
        attributes=span_kind(
            OpenInferenceSpanKindValues.EVALUATOR
        )
    ) as span:
        set_input(span, query, mime_type="text/plain")
        set_attributes(
            span,
            {
                "evaluation.type": "retrieval",
                "evaluation.top_k": top_k,
                "benchmark.run_id": benchmark_run_id,
                "benchmark.query_id": benchmark_query_id,
            }
        )
        strategy_results = []

        for strategy in StrategyRegistry.COLLECTIONS:
            try:
                start_time = time.perf_counter()

                results = retrieve(
                    query=query,
                    strategy=strategy,
                    top_k=top_k,
                    telemetry_attributes={
                        "benchmark.run_id": benchmark_run_id,
                        "benchmark.query_id":
                            benchmark_query_id,
                    }
                )

                retrieval_time_ms = round(
                    (
                        time.perf_counter()
                        - start_time
                    ) * 1000,
                    2
                )

                retrieved_chunks = [
                    RetrievedChunk(
                        text=result.payload["text"],
                        score=result.score,
                        source_file=result.payload["source_file"],
                        chunk_id=result.payload["chunk_id"]
                    )
                    for result in results
                ]
                scores = [
                    chunk.score
                    for chunk in retrieved_chunks
                ]

                strategy_result = StrategyEvaluationResult(
                    strategy=strategy,
                    status="success",
                    max_score=round(max(scores), 4)
                    if scores else None,
                    min_score=round(min(scores), 4)
                    if scores else None,
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

            strategy_results.append(strategy_result)

        response = RetrievalEvaluationResponse(
            query=query,
            results=strategy_results
        )
        successful_results = [
            result
            for result in strategy_results
            if result.status == "success"
        ]
        set_attributes(
            span,
            {
                "evaluation.strategy_count":
                    len(strategy_results),
                "evaluation.successful_strategy_count":
                    len(successful_results),
                "evaluation.failed_strategy_count":
                    len(strategy_results)
                    - len(successful_results),
            }
        )
        set_output(
            span,
            {
                "strategies": [
                    {
                        "strategy": result.strategy,
                        "status": result.status,
                        "average_score":
                            result.average_score,
                        "retrieval_time_ms":
                            result.retrieval_time_ms,
                    }
                    for result in strategy_results
                ]
            }
        )

        return response
