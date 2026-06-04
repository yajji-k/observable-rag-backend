from app.ingestion.embedder import (
    generate_embedding
)

from app.retrieval.qdrant_client import (
    client
)


def search(
    query: str,
    collection_name: str
):

    # Generate embedding for user query
    query_embedding = generate_embedding(
        query
    )

    # Perform semantic similarity search
    results = client.query_points(
        collection_name=collection_name,
        query=query_embedding,
        limit=3
    )

    return results.points