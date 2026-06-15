from functools import lru_cache

from sentence_transformers import SentenceTransformer


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    return SentenceTransformer(
        "BAAI/bge-small-en-v1.5"
    )


def generate_embedding(text: str):

    # Convert text into semantic vector embedding
    embedding = get_embedding_model().encode(text)

    # Qdrant expects standard Python lists
    return embedding.tolist()
