from app.services.ingestion.embedder import (
    generate_embedding
)

from app.infrastructure.vector_store.qdrant import (
    client
)

from app.services.retrieval.strategy_registry import (
    StrategyRegistry
)


def collection_exists(
    collection_name: str
) -> bool:

    return client.collection_exists(
        collection_name=collection_name
    )


def retrieve(
    query: str,
    strategy: str,
    top_k: int = 3
) -> list:

    # Fetch collection name for required strategy
    collection_name = (
        StrategyRegistry.get_collection_name(
            strategy=strategy
        )
    )

    # Validate collection exists
    if not collection_exists(
        collection_name
    ):
        raise ValueError(
            f"Collection '{collection_name}' does not exist"
        )

    # Generate embedding for user query
    query_embedding = (
        generate_embedding(
            query
        )
    )

    # Perform similarity search
    results = client.query_points(
        collection_name=collection_name,
        query=query_embedding,
        limit=top_k
    )

    return results.points
