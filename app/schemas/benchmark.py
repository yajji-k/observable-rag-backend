from pydantic import BaseModel


class BenchmarkQuery(BaseModel):
    id: str
    query: str
    expected_document: str
    expected_topics: list[str]

class BenchmarkResult(BaseModel):
    query_id: str
    query: str
    strategy: str

    max_score: float | None = None
    min_score: float | None = None
    average_score: float | None = None

    retrieval_time_ms: float | None = None
    
    rank: int | None = None

    hit_at_1: bool = False
    hit_at_3: bool = False
    hit_at_5: bool = False


class StrategyBenchmarkStats(BaseModel):
    strategy: str

    average_score: float
    average_latency_ms: float

    recall_at_1: float
    recall_at_3: float
    recall_at_5: float

    precision_at_1: float
    precision_at_3: float
    precision_at_5: float

    ndcg_at_1: float
    ndcg_at_3: float
    ndcg_at_5: float

    mrr: float

    wins: int
    win_percentage: float

class BenchmarkSummary(BaseModel):
    total_queries: int
    strategy_results: list[StrategyBenchmarkStats]
