from openinference.semconv.trace import OpenInferenceSpanKindValues

from app.observability.tracing import (
    set_attributes,
    set_input,
    set_output,
    span_kind,
    tracer,
)
from app.schemas.benchmark import (
    BenchmarkQuery,
    BenchmarkResult,
)
from app.services.evaluation.retrieval_evaluator import (
    run_retrieval_evaluation,
)


class BenchmarkRunner:
    def __init__(self, top_k: int = 5):
        self.top_k = top_k

    def run(
        self,
        benchmark_queries: list[BenchmarkQuery],
        run_id: str | None = None
    ) -> list[BenchmarkResult]:

        benchmark_results: list[BenchmarkResult] = []

        for benchmark_query in benchmark_queries:
            with tracer.start_as_current_span(
                "benchmark.query",
                attributes=span_kind(
                    OpenInferenceSpanKindValues.EVALUATOR
                )
            ) as span:
                set_input(
                    span,
                    {
                        "query": benchmark_query.query,
                        "expected_topics":
                            benchmark_query.expected_topics,
                    }
                )
                set_attributes(
                    span,
                    {
                        "benchmark.run_id": run_id,
                        "benchmark.query_id":
                            benchmark_query.id,
                        "benchmark.top_k": self.top_k,
                    }
                )

                evaluation_response = (
                    run_retrieval_evaluation(
                        query=benchmark_query.query,
                        top_k=self.top_k,
                        benchmark_run_id=run_id,
                        benchmark_query_id=
                            benchmark_query.id,
                    )
                )

                query_results = []

                for strategy_result in (
                    evaluation_response.results
                ):
                    if strategy_result.status != "success":
                        continue

                    result = BenchmarkResult(
                        query_id=benchmark_query.id,
                        query=benchmark_query.query,
                        strategy=strategy_result.strategy,
                        max_score=strategy_result.max_score,
                        min_score=strategy_result.min_score,
                        average_score=
                            strategy_result.average_score,
                        retrieval_time_ms=
                            strategy_result.retrieval_time_ms,
                    )
                    query_results.append(result)
                    benchmark_results.append(result)

                winner = max(
                    query_results,
                    key=lambda result: result.average_score
                ) if query_results else None

                set_attributes(
                    span,
                    {
                        "benchmark.winning_strategy":
                            winner.strategy
                            if winner else None,
                        "benchmark.winning_score":
                            winner.average_score
                            if winner else None,
                        "benchmark.successful_strategy_count":
                            len(query_results),
                    }
                )
                set_output(
                    span,
                    {
                        "winning_strategy":
                            winner.strategy
                            if winner else None,
                        "strategy_results": [
                            result.model_dump()
                            for result in query_results
                        ],
                    }
                )

        return benchmark_results
