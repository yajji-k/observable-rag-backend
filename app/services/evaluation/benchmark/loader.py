import json
from pathlib import Path

from app.schemas.benchmark import BenchmarkQuery


class BenchmarkLoader:
    def __init__(self, dataset_path: str | Path):
        self.dataset_path = Path(dataset_path)

    def load_dataset(self) -> list[BenchmarkQuery]:
        """
        Load benchmark queries from a JSON dataset file and
        validate them using the BenchmarkQuery model.
        """

        if not self.dataset_path.exists():
            raise FileNotFoundError(
                f"Benchmark dataset not found: {self.dataset_path}"
            )

        with open(self.dataset_path, "r", encoding="utf-8") as file:
            raw_data = json.load(file)

        if not isinstance(raw_data, list):
            raise ValueError(
                "Benchmark dataset must be a JSON array of queries."
            )

        return [
            BenchmarkQuery.model_validate(query)
            for query in raw_data
        ]
