from dataclasses import dataclass


@dataclass(frozen=True)
class RerankerConfig:
    enabled: bool
    model: str
    candidate_count: int
    final_top_k: int


@dataclass(frozen=True)
class RerankingResult:
    chunks: list
    scores: list[float]
    elapsed_ms: float

    @property
    def average_score(self) -> float | None:
        if not self.scores:
            return None

        return round(
            sum(self.scores) / len(self.scores),
            4
        )
