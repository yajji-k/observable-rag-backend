from qdrant_client import QdrantClient

from qdrant_client.models import (
    VectorParams,
    Distance,
    PointStruct
)

from app.config import (
    QDRANT_HOST,
    QDRANT_PORT,
)


# Initialize Qdrant client
client = QdrantClient(
    host=QDRANT_HOST,
    port=QDRANT_PORT
)


def create_collection(COLLECTION_NAME):

    # Recreate collection for fresh ingestion
    client.recreate_collection(

        collection_name=COLLECTION_NAME,

        vectors_config=VectorParams(

            # Must match embedding model output dimension
            size=384,

            distance=Distance.COSINE
        )
    )

    print("Collection created!")


def insert_documents(
    COLLECTION_NAME,
    chunks,
    embeddings
):

    points = []

    # Combine chunks with their embeddings
    for idx, (chunk, embedding) in enumerate(
        zip(chunks, embeddings)
    ):

        point = PointStruct(

            id=idx,

            vector=embedding,

            payload={

                # Original chunk text
                "text": chunk,

                # Metadata for future filtering
                "domain": "general"
            }
        )

        points.append(point)

    # Insert all points into Qdrant
    client.upsert(

        collection_name=COLLECTION_NAME,

        points=points
    )

    print("Chunks inserted!")