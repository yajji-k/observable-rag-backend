from typing import List

from .base_chunker import BaseChunker


class CharacterChunker(BaseChunker):

    def __init__(
        self,
        chunk_size: int = 500,
        overlap: int = 100
    ):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> List[str]:

        chunks = []

        start = 0

        while start < len(text):

            end = start + self.chunk_size

            chunk = text[start:end]

            chunks.append(chunk)

            start += self.chunk_size - self.overlap

        return chunks
