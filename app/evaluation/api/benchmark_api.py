from fastapi import APIRouter
from app.evaluation.benchmark.benchmark_runner import BenchmarkRunner
from app.evaluation.benchmark.benchmark_loader import BenchmarkLoader
from app.evaluation.benchmark.benchmark_aggregator import BenchmarkAggregator


benchmark_router = APIRouter()

@benchmark_router.post("/benchmark/default")
def run_benchmark():

    loader = BenchmarkLoader(
        "app/evaluation/benchmark/benchmark_dataset.json"
    )

    queries = loader.load_dataset()

    runner = BenchmarkRunner()

    results = runner.run(queries)

    aggregator = BenchmarkAggregator()

    summary = aggregator.aggregate(results)

    return summary