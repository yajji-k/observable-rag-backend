from typing import Any

from sentence_transformers import CrossEncoder

from app.core.config import RERANKER_BGE_MODEL_NAME
from app.services.reranking.base import BaseReranker


class BGEReranker(BaseReranker):

    def __init__(
        self,
        model_name: str = RERANKER_BGE_MODEL_NAME
    ):
        self._model_name = model_name
        self._model = CrossEncoder(model_name)

    @property
    def model_name(self) -> str:
        return self._model_name

    def rerank(
        self,
        query: str,
        chunks: list[Any],
        top_k: int
    ) -> list[Any]:
        if not chunks:
            return []

        pairs = [
            [
                query,
                str((chunk.payload or {}).get("text", ""))
            ]
            for chunk in chunks
        ]
        scores = self._model.predict(pairs)

        scored_chunks = []
        for chunk, score in zip(chunks, scores):
            payload = chunk.payload or {}
            payload["vector_score"] = float(chunk.score)
            payload["rerank_score"] = float(score)
            chunk.payload = payload
            scored_chunks.append(chunk)

        scored_chunks.sort(
            key=lambda chunk: (
                chunk.payload or {}
            ).get("rerank_score", float("-inf")),
            reverse=True
        )

        return scored_chunks[:top_k]
