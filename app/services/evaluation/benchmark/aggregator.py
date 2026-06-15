from collections import defaultdict

from openinference.semconv.trace import OpenInferenceSpanKindValues

from app.observability.tracing import (
    set_attributes,
    set_input,
    set_output,
    span_kind,
    tracer,
)
from app.schemas.benchmark import (
    BenchmarkResult,
    BenchmarkSummary,
    StrategyBenchmarkStats,
)


class BenchmarkAggregator:

    def aggregate(
        self,
        results: list[BenchmarkResult],
        run_id: str | None = None
    ) -> BenchmarkSummary:
        with tracer.start_as_current_span(
            "benchmark.aggregate",
            attributes=span_kind(
                OpenInferenceSpanKindValues.EVALUATOR
            )
        ) as span:
            set_input(
                span,
                {
                    "result_count": len(results)
                }
            )
            set_attributes(
                span,
                {
                    "benchmark.run_id": run_id,
                    "benchmark.result_count": len(results),
                }
            )

            summary = self._aggregate(results)
            set_attributes(
                span,
                {
                    "benchmark.total_queries":
                        summary.total_queries,
                }
            )
            set_output(span, summary.model_dump())

            return summary

    def _aggregate(
        self,
        results: list[BenchmarkResult]
    ) -> BenchmarkSummary:
        if not results:
            return BenchmarkSummary(
                total_queries=0,
                strategy_results=[]
            )

        strategy_groups: dict[str, list[BenchmarkResult]] = defaultdict(list)

        for result in results:
            strategy_groups[result.strategy].append(result)

        strategy_stats: list[StrategyBenchmarkStats] = []

        # --------------------------------------------------
        # Calculate winners
        # --------------------------------------------------

        wins: dict[str, int] = defaultdict(int)

        query_groups: dict[str, list[BenchmarkResult]] = defaultdict(list)

        for result in results:
            query_groups[result.query_id].append(result)

        for query_results in query_groups.values():

            winner = max(
                query_results,
                key=lambda r: r.average_score
            )

            wins[winner.strategy] += 1

        total_queries = len(query_groups)

        # --------------------------------------------------
        # Aggregate strategy metrics
        # --------------------------------------------------

        for strategy, strategy_results in strategy_groups.items():

            average_score = (
                sum(r.average_score for r in strategy_results)
                / len(strategy_results)
            )

            average_latency_ms = (
                sum(r.retrieval_time_ms for r in strategy_results)
                / len(strategy_results)
            )

            strategy_wins = wins.get(strategy, 0)

            win_percentage = (
                strategy_wins / total_queries * 100
            )

            strategy_stats.append(
                StrategyBenchmarkStats(
                    strategy=strategy,
                    average_score=round(average_score, 4),
                    average_latency_ms=round(
                        average_latency_ms,
                        2
                    ),
                    wins=strategy_wins,
                    win_percentage=round(
                        win_percentage,
                        2
                    ),
                )
            )

        return BenchmarkSummary(
            total_queries=total_queries,
            strategy_results=strategy_stats,
        )
