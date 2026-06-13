from app.evaluation.models.benchmark_models import (
    BenchmarkQuery,
    BenchmarkResult,
)
from app.evaluation.retriever.retrieval_evaluator import (
    run_retrieval_evaluation,
)


class BenchmarkRunner:
    def __init__(self, top_k: int = 5):
        self.top_k = top_k

    def run(
        self,
        benchmark_queries: list[BenchmarkQuery]
    ) -> list[BenchmarkResult]:

        benchmark_results: list[BenchmarkResult] = []

        for benchmark_query in benchmark_queries:

            evaluation_response = run_retrieval_evaluation(
                query=benchmark_query.query,
                top_k=self.top_k
            )

            for strategy_result in evaluation_response.results:

                # Skip failed strategies
                if strategy_result.status != "success":
                    continue

                benchmark_results.append(
                    BenchmarkResult(
                        query_id=benchmark_query.id,
                        query=benchmark_query.query,

                        strategy=strategy_result.strategy,

                        max_score=strategy_result.max_score,
                        min_score=strategy_result.min_score,
                        average_score=strategy_result.average_score,

                        retrieval_time_ms=strategy_result.retrieval_time_ms,
                    )
                )

        return benchmark_results