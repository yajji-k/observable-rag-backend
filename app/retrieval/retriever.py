from app.ingestion.embedder import (
    generate_embedding
)

from app.retrieval.qdrant_client import (
    client
)

from app.config import COLLECTION_NAME


def search(query: str):

    # Generate embedding for user query
    query_embedding = generate_embedding(
        query
    )

    # Perform semantic similarity search
    results = client.query_points(

        collection_name=COLLECTION_NAME,

        query=query_embedding,

        # Number of relevant chunks to retrieve
        limit=3
    )

    return results.points