from pydantic import BaseModel


class RetrievalEvaluationRequest(BaseModel):
    query: str
    top_k: int = 3


class RetrievedChunk(BaseModel):
    text: str
    score: float
    source_file: str
    chunk_id: int


class StrategyEvaluationResult(BaseModel):
    strategy: str
    status: str

    max_score: float | None = None
    min_score: float | None = None
    average_score: float | None = None

    retrieval_time_ms: float | None = None

    retrieved_chunks: list[RetrievedChunk]

class RetrievalEvaluationResponse(BaseModel):
    query: str
    results: list[StrategyEvaluationResult]