from collections import defaultdict

from openinference.semconv.trace import (
    OpenInferenceSpanKindValues
)

import math
from collections import defaultdict

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

            set_output(
                span,
                summary.model_dump()
            )

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

        # ------------------------------------------
        # Group by strategy
        # ------------------------------------------

        strategy_groups: dict[
            str,
            list[BenchmarkResult]
        ] = defaultdict(list)

        for result in results:
            strategy_groups[
                result.strategy
            ].append(result)

        # ------------------------------------------
        # Group by query
        # ------------------------------------------

        query_groups: dict[
            str,
            list[BenchmarkResult]
        ] = defaultdict(list)

        for result in results:
            query_groups[
                result.query_id
            ].append(result)

        # ------------------------------------------
        # Determine winners
        # ------------------------------------------

        wins: dict[str, int] = defaultdict(int)

        for query_results in query_groups.values():

            winner = max(
                query_results,
                key=lambda r: (
                    r.average_score
                    if r.average_score is not None
                    else 0
                )
            )

            wins[winner.strategy] += 1

        total_queries = len(query_groups)

        strategy_stats: list[
            StrategyBenchmarkStats
        ] = []

        # ------------------------------------------
        # Aggregate metrics
        # ------------------------------------------

        for (
            strategy,
            strategy_results
        ) in strategy_groups.items():

            if not strategy_results:
                continue

            # ----------------------------
            # Score Metrics
            # ----------------------------

            valid_scores = [
                r.average_score
                for r in strategy_results
                if r.average_score is not None
            ]

            average_score = (
                sum(valid_scores)
                / len(valid_scores)
                if valid_scores
                else 0.0
            )

            # ----------------------------
            # Latency Metrics
            # ----------------------------

            valid_latencies = [
                r.retrieval_time_ms
                for r in strategy_results
                if r.retrieval_time_ms is not None
            ]

            average_latency_ms = (
                sum(valid_latencies)
                / len(valid_latencies)
                if valid_latencies
                else 0.0
            )

            # ----------------------------
            # Recall Metrics
            # ----------------------------

            recall_at_1 = (
                sum(
                    1
                    for r in strategy_results
                    if r.hit_at_1
                )
                / len(strategy_results)
            )

            recall_at_3 = (
                sum(
                    1
                    for r in strategy_results
                    if r.hit_at_3
                )
                / len(strategy_results)
            )

            recall_at_5 = (
                sum(
                    1
                    for r in strategy_results
                    if r.hit_at_5
                )
                / len(strategy_results)
            )

            # ----------------------------
            # MRR
            # ----------------------------

            mrr = (
                sum(
                    (
                        1 / r.rank
                    )
                    if (
                        r.rank is not None
                        and r.rank > 0
                    )
                    else 0
                    for r in strategy_results
                )
                / len(strategy_results)
            )
            
            # ----------------------------
            # Precision@K
            # ----------------------------

            precision_at_1 = (
                sum(
                    (
                        1.0
                        if (
                            r.rank is not None
                            and r.rank <= 1
                        )
                        else 0.0
                    )
                    for r in strategy_results
                )
                / len(strategy_results)
            )

            precision_at_3 = (
                sum(
                    (
                        1 / 3
                        if (
                            r.rank is not None
                            and r.rank <= 3
                        )
                        else 0.0
                    )
                    for r in strategy_results
                )
                / len(strategy_results)
            )

            precision_at_5 = (
                sum(
                    (
                        1 / 5
                        if (
                            r.rank is not None
                            and r.rank <= 5
                        )
                        else 0.0
                    )
                    for r in strategy_results
                )
                / len(strategy_results)
            )
            
            # ----------------------------
            # NDCG@K
            # ----------------------------

            ndcg_at_1 = (
                sum(
                    (
                        1.0 / math.log2(
                            r.rank + 1
                        )
                    )
                    if (
                        r.rank is not None
                        and r.rank <= 1
                    )
                    else 0.0
                    for r in strategy_results
                )
                / len(strategy_results)
            )

            ndcg_at_3 = (
                sum(
                    (
                        1.0 / math.log2(
                            r.rank + 1
                        )
                    )
                    if (
                        r.rank is not None
                        and r.rank <= 3
                    )
                    else 0.0
                    for r in strategy_results
                )
                / len(strategy_results)
            )

            ndcg_at_5 = (
                sum(
                    (
                        1.0 / math.log2(
                            r.rank + 1
                        )
                    )
                    if (
                        r.rank is not None
                        and r.rank <= 5
                    )
                    else 0.0
                    for r in strategy_results
                )
                / len(strategy_results)
            )

            # ----------------------------
            # Win Stats
            # ----------------------------

            strategy_wins = wins.get(
                strategy,
                0
            )

            win_percentage = (
                (
                    strategy_wins
                    / total_queries
                )
                * 100
                if total_queries > 0
                else 0
            )

            strategy_stats.append(
                StrategyBenchmarkStats(
                    strategy=strategy,

                    average_score=round(
                        average_score,
                        4
                    ),

                    average_latency_ms=round(
                        average_latency_ms,
                        2
                    ),

                    recall_at_1=round(
                        recall_at_1,
                        4
                    ),

                    recall_at_3=round(
                        recall_at_3,
                        4
                    ),

                    recall_at_5=round(
                        recall_at_5,
                        4
                    ),

                    mrr=round(
                        mrr,
                        4
                    ),
                    
                    precision_at_1=round(
                        precision_at_1,
                        4
                    ),

                    precision_at_3=round(
                        precision_at_3,
                        4
                    ),

                    precision_at_5=round(
                        precision_at_5,
                        4
                    ),

                    ndcg_at_1=round(
                        ndcg_at_1,
                        4
                    ),

                    ndcg_at_3=round(
                        ndcg_at_3,
                        4
                    ),

                    ndcg_at_5=round(
                        ndcg_at_5,
                        4
                    ),

                    wins=strategy_wins,

                    win_percentage=round(
                        win_percentage,
                        2
                    ),
                )
            )

        # ------------------------------------------
        # Sort Best Strategy First
        # ------------------------------------------

        strategy_stats.sort(
            key=lambda s: (
                s.mrr,
                s.ndcg_at_5,
                s.recall_at_1,
                s.recall_at_3,
                s.recall_at_5,
            ),
            reverse=True
        )

        return BenchmarkSummary(
            total_queries=total_queries,
            strategy_results=strategy_stats,
        )