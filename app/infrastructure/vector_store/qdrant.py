from qdrant_client import QdrantClient
import uuid
from qdrant_client.models import (
    VectorParams,
    Distance,
    PointStruct
)

from app.core.config import (
    QDRANT_HOST,
    QDRANT_PORT,
)


# Initialize Qdrant client
client = QdrantClient(
    host=QDRANT_HOST,
    port=QDRANT_PORT
)


def create_collection(collection_name: str):

    collections = client.get_collections().collections

    if any(
        collection.name == collection_name
        for collection in collections
    ):
        print(f"Collection '{collection_name}' already exists.")
        return

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=384,
            distance=Distance.COSINE
        )
    )

    print(f"Collection '{collection_name}' created!")


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
            id=str(uuid.uuid4()),

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
    
    
def delete_collection(collection_name: str):

    collections = client.get_collections().collections

    if not any(
        collection.name == collection_name
        for collection in collections
    ):
        print(
            f"Collection '{collection_name}' does not exist."
        )
        return

    client.delete_collection(
        collection_name=collection_name
    )

    print(
        f"Collection '{collection_name}' deleted."
    )
    
def delete_all_rag_collections():

    collections = client.get_collections().collections

    for collection in collections:

        if collection.name.startswith(
            "rag_documents_"
        ):

            client.delete_collection(
                collection_name=collection.name
            )

            print(
                f"Deleted: {collection.name}"
            )
