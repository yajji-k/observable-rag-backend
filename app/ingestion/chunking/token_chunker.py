from typing import List

import tiktoken

from .base_chunker import BaseChunker


class TokenChunker(BaseChunker):

    def __init__(
        self,
        chunk_size: int = 500,
        overlap: int = 100,
        encoding_name: str = "cl100k_base"
    ):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.encoding = tiktoken.get_encoding(
            encoding_name
        )

    def chunk(self, text: str) -> List[str]:
        tokens = self.encoding.encode(text)
        chunks = []
        start = 0

        while start < len(tokens):
            end = start + self.chunk_size
            chunk_tokens = tokens[start:end]
            chunk_text = self.encoding.decode(chunk_tokens)

            chunks.append(chunk_text)
            start += (self.chunk_size - self.overlap)

        return chunks