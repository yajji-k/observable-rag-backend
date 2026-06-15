from pydantic import BaseModel, Field


class BenchmarkQuery(BaseModel):
    id: str
    query: str
    expected_topics: list[str] = Field(default_factory=list)


class BenchmarkResult(BaseModel):
    query_id: str
    query: str
    strategy: str

    max_score: float
    min_score: float
    average_score: float

    retrieval_time_ms: float


class StrategyBenchmarkStats(BaseModel):
    strategy: str

    average_score: float
    average_latency_ms: float

    wins: int
    win_percentage: float


class BenchmarkSummary(BaseModel):
    total_queries: int
    strategy_results: list[StrategyBenchmarkStats]
