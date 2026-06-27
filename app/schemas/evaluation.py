from pydantic import BaseModel


class RetrievalEvaluationRequest(BaseModel):
    query: str
    top_k: int = 3
    reranking_enabled: bool | None = None
    reranker_model: str | None = None
    candidate_count: int | None = None


class RetrievedChunk(BaseModel):
    text: str
    score: float
    vector_score: float | None = None
    rerank_score: float | None = None
    source_file: str
    chunk_id: int


class StrategyEvaluationResult(BaseModel):
    strategy: str
    status: str

    max_score: float | None = None
    min_score: float | None = None
    average_score: float | None = None
    average_rerank_score: float | None = None

    retrieval_time_ms: float | None = None

    retrieved_chunks: list[RetrievedChunk]

class RetrievalEvaluationResponse(BaseModel):
    query: str
    results: list[StrategyEvaluationResult]
