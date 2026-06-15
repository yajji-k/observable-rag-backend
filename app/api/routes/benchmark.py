from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter
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
def run_benchmark():
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
                }
            )

            loader = BenchmarkLoader(
                DEFAULT_DATASET_PATH
            )
            queries = loader.load_dataset()

            runner = BenchmarkRunner()
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
