from app.ingestion.chunking.token_chunker import TokenChunker

from .character_chunker import CharacterChunker
from .recursive_chunker import RecursiveChunker


class ChunkerFactory:

    _chunkers = {
        "character": CharacterChunker,
        "recursive": RecursiveChunker,
        "token": TokenChunker
    }

    @classmethod
    def get_available_strategies(cls):
        return list(cls._chunkers.keys())

    @classmethod
    def create(
        cls,
        strategy: str,
        chunk_size: int = 500,
        overlap: int = 100
    ):
        chunker_class = cls._chunkers.get(strategy)

        if chunker_class is None:
            available = ", ".join(cls._chunkers.keys())

            raise ValueError(
                f"Unsupported chunking strategy '{strategy}'. "
                f"Available strategies: {available}"
            )

        return chunker_class