from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, Query
from openinference.instrumentation import using_attributes
from openinference.semconv.trace import OpenInferenceSpanKindValues

from app.observability.tracing import (
    set_attributes,
    set_input,
    set_output,
    span_kind,
    tracer,
)
from app.services.evaluation.benchmark.aggregator import BenchmarkAggregator
from app.services.evaluation.benchmark.loader import BenchmarkLoader
from app.services.evaluation.benchmark.runner import BenchmarkRunner


benchmark_router = APIRouter()
DEFAULT_DATASET_PATH = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "benchmark_dataset.json"
)

@benchmark_router.post("/benchmark/default")
def run_benchmark(
    reranking_enabled: bool | None = Query(
        default=None
    ),
    reranker_model: str | None = Query(
        default=None
    ),
    candidate_count: int | None = Query(
        default=None
    )
):
    run_id = str(uuid4())

    with using_attributes(
        session_id=run_id,
        metadata={
            "benchmark_dataset":
                DEFAULT_DATASET_PATH.name
        },
        tags=["benchmark", "retrieval-evaluation"]
    ):
        with tracer.start_as_current_span(
            "benchmark.run",
            attributes=span_kind(
                OpenInferenceSpanKindValues.CHAIN
            )
        ) as span:
            set_input(
                span,
                {
                    "dataset": DEFAULT_DATASET_PATH.name
                }
            )
            set_attributes(
                span,
                {
                    "benchmark.run_id": run_id,
                    "benchmark.dataset":
                        DEFAULT_DATASET_PATH.name,
                    "benchmark.reranking_enabled":
                        reranking_enabled,
                    "benchmark.reranker_model":
                        reranker_model,
                    "benchmark.candidate_count":
                        candidate_count,
                }
            )

            loader = BenchmarkLoader(
                DEFAULT_DATASET_PATH
            )
            queries = loader.load_dataset()

            runner = BenchmarkRunner(
                reranking_enabled=reranking_enabled,
                reranker_model=reranker_model,
                candidate_count=candidate_count
            )
            results = runner.run(
                queries,
                run_id=run_id
            )

            aggregator = BenchmarkAggregator()
            summary = aggregator.aggregate(
                results,
                run_id=run_id
            )

            set_attributes(
                span,
                {
                    "benchmark.query_count":
                        len(queries),
                    "benchmark.result_count":
                        len(results),
                }
            )
            set_output(span, summary.model_dump())

            return summary
