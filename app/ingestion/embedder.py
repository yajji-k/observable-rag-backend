from sentence_transformers import ( SentenceTransformer )


# Load embedding model once during application startup
model = SentenceTransformer(
    "BAAI/bge-small-en-v1.5"
)


def generate_embedding(text: str):

    # Convert text into semantic vector embedding
    embedding = model.encode(text)

    # Qdrant expects standard Python lists
    return embedding.tolist()