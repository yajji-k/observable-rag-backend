import re
import numpy as np

from app.services.ingestion.embedder import generate_embedding

from .base_chunker import BaseChunker


def split_sentences(text: str):
    """
    Split document text into individual sentences.
    """

    return re.split(
        r'(?<=[.!?])\s+',
        text
    )


def cosine_similarity(vector1, vector2):
    """
    Measure semantic similarity between two embeddings.

    Returns:
        1.0  -> identical meaning
        0.0  -> unrelated meaning
       -1.0  -> opposite meaning
    """

    vec1 = np.array(vector1)
    vec2 = np.array(vector2)

    return np.dot(vec1, vec2) / (
        np.linalg.norm(vec1)
        * np.linalg.norm(vec2)
    )


class SemanticChunker(BaseChunker):

    def __init__(
        self,
        chunk_size: int = 500,
        overlap: int = 100,
        similarity_threshold: float = 0.75
    ):
        """
        chunk_size and overlap are included to maintain
        a consistent constructor signature across all
        chunking strategies.

        similarity_threshold controls when a new
        semantic chunk should be created.
        """

        self.chunk_size = chunk_size
        self.overlap = overlap
        self.similarity_threshold = similarity_threshold

    def chunk(self, text: str):
        """
        Create chunks based on semantic similarity.

        Workflow:

            Document
                ↓
            Sentence Split
                ↓
            Sentence Embeddings
                ↓
            Cosine Similarity
                ↓
            Topic Change Detection
                ↓
            Semantic Chunks
        """

        # Split document into sentences
        sentences = split_sentences(text)

        # Handle empty documents
        if not sentences:
            return []

        # Generate embedding for each sentence
        embeddings = [
            generate_embedding(sentence)
            for sentence in sentences
        ]

        # Store final chunks
        chunks = []

        # Start first chunk with first sentence
        current_chunk = [sentences[0]]

        # Compare neighboring sentences
        for i in range(1, len(sentences)):

            similarity = cosine_similarity(
                embeddings[i - 1],
                embeddings[i]
            )

            # Topic change detected
            if similarity < self.similarity_threshold:

                # Save previous chunk
                chunks.append(
                    " ".join(current_chunk)
                )

                # Start new chunk
                current_chunk = [
                    sentences[i]
                ]

            else:

                # Continue current chunk
                current_chunk.append(
                    sentences[i]
                )

        # Save final chunk
        chunks.append(
            " ".join(current_chunk)
        )

        return chunks
