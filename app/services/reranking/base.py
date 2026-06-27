from abc import ABC, abstractmethod
from typing import Any


class BaseReranker(ABC):

    @property
    @abstractmethod
    def model_name(self) -> str:
        pass

    @abstractmethod
    def rerank(
        self,
        query: str,
        chunks: list[Any],
        top_k: int
    ) -> list[Any]:
        pass
