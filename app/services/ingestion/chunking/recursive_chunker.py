from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter

from .base_chunker import BaseChunker


class RecursiveChunker(BaseChunker):

    def __init__(
        self,
        chunk_size: int = 500,
        overlap: int = 100
    ):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            separators=[
                "\n\n",
                "\n",
                ". ",
                " ",
                ""
            ]
        )

    def chunk(self, text: str) -> List[str]:

        chunks = self.splitter.split_text(text)

        return chunks
