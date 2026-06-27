import time
from typing import Any

from app.services.reranking.models import (
    RerankerConfig,
    RerankingResult,
)
from app.services.reranking.registry import RerankerRegistry


class RerankingService:

    def __init__(
        self,
        config: RerankerConfig
    ):
        self.config = config

    @property
    def model_name(self) -> str:
        reranker = RerankerRegistry.get(
            self.config.model
        )

        return reranker.model_name

    def rerank(
        self,
        query: str,
        chunks: list[Any],
        top_k: int
    ) -> RerankingResult:
        started_at = time.perf_counter()
        reranker = RerankerRegistry.get(
            self.config.model
        )
        reranked_chunks = reranker.rerank(
            query=query,
            chunks=chunks,
            top_k=top_k
        )
        scores = [
            float(
                (chunk.payload or {}).get(
                    "rerank_score"
                )
            )
            for chunk in reranked_chunks
            if (
                chunk.payload or {}
            ).get("rerank_score") is not None
        ]

        return RerankingResult(
            chunks=reranked_chunks,
            scores=scores,
            elapsed_ms=round(
                (time.perf_counter() - started_at)
                * 1000,
                2
            )
        )
