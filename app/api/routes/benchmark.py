from pathlib import Path

from fastapi import APIRouter

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

    loader = BenchmarkLoader(
        DEFAULT_DATASET_PATH
    )

    queries = loader.load_dataset()

    runner = BenchmarkRunner()

    results = runner.run(queries)

    aggregator = BenchmarkAggregator()

    summary = aggregator.aggregate(results)

    return summary
