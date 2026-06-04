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
    chunk_records,
    embeddings
):

    points = []

    # Combine chunk data with its embedding
    # and convert it into a Qdrant PointStruct
    for idx, (record, embedding) in enumerate(
        zip(chunk_records, embeddings)
    ):

        point = PointStruct(

            # Unique point id within collection
            id=idx,

            # Vector embedding generated from chunk text
            vector=embedding,

            # Metadata stored alongside vector
            payload={

                # Original chunk content
                "text": record["text"],

                # Source PDF filename
                "source_file": record["source_file"],

                # Position of chunk within document
                "chunk_id": record["chunk_id"],

                # Chunking strategy used during ingestion
                "chunk_strategy": record["chunk_strategy"],

                # Placeholder metadata for future filtering
                "domain": "general"
            }
        )

        points.append(point)

    # Insert all vectors and metadata into Qdrant
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )

    print("Chunks inserted!")