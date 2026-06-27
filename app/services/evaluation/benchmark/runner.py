from pathlib import Path

from openinference.semconv.trace import (
    OpenInferenceSpanKindValues
)

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

    def __init__(
        self,
        top_k: int = 5,
        reranking_enabled: bool | None = None,
        reranker_model: str | None = None,
        candidate_count: int | None = None
    ):
        self.top_k = top_k
        self.reranking_enabled = reranking_enabled
        self.reranker_model = reranker_model
        self.candidate_count = candidate_count

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
                        "query":
                            benchmark_query.query,

                        "expected_topics":
                            benchmark_query.expected_topics,

                        "expected_document":
                            benchmark_query.expected_document,
                    }
                )

                set_attributes(
                    span,
                    {
                        "benchmark.run_id":
                            run_id,

                        "benchmark.query_id":
                            benchmark_query.id,

                        "benchmark.top_k":
                            self.top_k,
                        "benchmark.reranking_enabled":
                            self.reranking_enabled,
                        "benchmark.reranker_model":
                            self.reranker_model,
                        "benchmark.candidate_count":
                            self.candidate_count,
                    }
                )

                evaluation_response = (
                    run_retrieval_evaluation(
                        query=benchmark_query.query,
                        top_k=self.top_k,
                        benchmark_run_id=run_id,
                        benchmark_query_id=
                            benchmark_query.id,
                        reranking_enabled=
                            self.reranking_enabled,
                        reranker_model=
                            self.reranker_model,
                        candidate_count=
                            self.candidate_count,
                    )
                )

                query_results: list[
                    BenchmarkResult
                ] = []

                for strategy_result in (
                    evaluation_response.results
                ):

                    if (
                        strategy_result.status
                        != "success"
                    ):
                        continue

                    # ----------------------------------
                    # Find expected document rank
                    # ----------------------------------

                    rank = None

                    for index, chunk in enumerate(
                        strategy_result.retrieved_chunks,
                        start=1
                    ):

                        source_file = Path(
                            chunk.source_file
                        ).name

                        if (
                            source_file
                            ==
                            benchmark_query.expected_document
                        ):
                            rank = index
                            break

                    hit_at_1 = (
                        rank is not None
                        and rank <= 1
                    )

                    hit_at_3 = (
                        rank is not None
                        and rank <= 3
                    )

                    hit_at_5 = (
                        rank is not None
                        and rank <= 5
                    )

                    strategy_name = (
                        f"{strategy_result.strategy}+rerank"
                        if self.reranking_enabled
                        else strategy_result.strategy
                    )

                    result = BenchmarkResult(
                        query_id=
                            benchmark_query.id,

                        query=
                            benchmark_query.query,

                        strategy=strategy_name,

                        max_score=
                            strategy_result.max_score,

                        min_score=
                            strategy_result.min_score,

                        average_score=
                            strategy_result.average_score,

                        average_rerank_score=
                            strategy_result.average_rerank_score,

                        retrieval_time_ms=
                            strategy_result.retrieval_time_ms,

                        rank=rank,

                        hit_at_1=hit_at_1,

                        hit_at_3=hit_at_3,

                        hit_at_5=hit_at_5,
                    )

                    query_results.append(result)
                    benchmark_results.append(result)

                # ----------------------------------
                # Current winner logic
                # (keep score-based for now)
                # ----------------------------------

                winner = (
                    max(
                        query_results,
                        key=lambda result:
                            result.average_score
                    )
                    if query_results
                    else None
                )

                best_rank = min(
                    (
                        r.rank
                        for r in query_results
                        if r.rank is not None
                    ),
                    default=None,
                )

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

                        "benchmark.expected_document_found":
                            any(
                                r.hit_at_5
                                for r in query_results
                            ),

                        "benchmark.best_rank":
                            best_rank,
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
                        ]
                    }
                )

        return benchmark_results
