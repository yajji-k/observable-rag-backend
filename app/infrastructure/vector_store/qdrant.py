from qdrant_client import QdrantClient
import uuid
from openinference.semconv.trace import OpenInferenceSpanKindValues
from qdrant_client.models import (
    VectorParams,
    Distance,
    PointStruct
)

from app.core.config import (
    QDRANT_HOST,
    QDRANT_PORT,
)
from app.observability.tracing import (
    set_attributes,
    set_input,
    set_output,
    span_kind,
    tracer,
)


# Initialize Qdrant client
client = QdrantClient(
    host=QDRANT_HOST,
    port=QDRANT_PORT
)


def create_collection(collection_name: str):
    with tracer.start_as_current_span(
        "qdrant.create_collection",
        attributes=span_kind(
            OpenInferenceSpanKindValues.TOOL
        )
    ) as span:
        set_attributes(
            span,
            {
                "qdrant.collection": collection_name,
                "qdrant.vector_size": 384,
                "qdrant.distance": "cosine",
            }
        )

        collections = client.get_collections().collections

        if any(
            collection.name == collection_name
            for collection in collections
        ):
            span.set_attribute(
                "qdrant.collection_created",
                False
            )
            print(f"Collection '{collection_name}' already exists.")
            return

        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=384,
                distance=Distance.COSINE
            )
        )

        span.set_attribute(
            "qdrant.collection_created",
            True
        )
        print(f"Collection '{collection_name}' created!")


def insert_documents(
    collection_name,
    chunk_records,
    embeddings
):
    with tracer.start_as_current_span(
        "qdrant.upsert_documents",
        attributes=span_kind(
            OpenInferenceSpanKindValues.TOOL
        )
    ) as span:
        set_attributes(
            span,
            {
                "qdrant.collection": collection_name,
                "qdrant.point_count": len(chunk_records),
            }
        )
        points = []

        for record, embedding in zip(
            chunk_records,
            embeddings
        ):
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload={
                    "text": record["text"],
                    "source_file": record["source_file"],
                    "document_name": record["source_file"],
                    "chunk_id": record["chunk_id"],
                    "chunk_strategy": record["chunk_strategy"]
                }
            )

            points.append(point)

        client.upsert(
            collection_name=collection_name,
            points=points
        )

        set_output(
            span,
            {
                "inserted_points": len(points)
            }
        )
        print("Chunks inserted!")


def query_points(
    collection_name: str,
    query_embedding: list[float],
    top_k: int
) -> list:
    with tracer.start_as_current_span(
        "qdrant.query_points",
        attributes=span_kind(
            OpenInferenceSpanKindValues.TOOL
        )
    ) as span:
        set_input(
            span,
            {
                "collection": collection_name,
                "top_k": top_k,
                "embedding_dimensions": len(query_embedding),
            }
        )

        results = client.query_points(
            collection_name=collection_name,
            query=query_embedding,
            limit=top_k
        ).points

        set_output(
            span,
            {
                "result_count": len(results)
            }
        )

        return results


def delete_collection(collection_name: str):
    with tracer.start_as_current_span(
        "qdrant.delete_collection",
        attributes=span_kind(
            OpenInferenceSpanKindValues.TOOL
        )
    ) as span:
        span.set_attribute(
            "qdrant.collection",
            collection_name
        )

        collections = client.get_collections().collections

        if not any(
            collection.name == collection_name
            for collection in collections
        ):
            span.set_attribute("qdrant.deleted", False)
            print(
                f"Collection '{collection_name}' does not exist."
            )
            return

        client.delete_collection(
            collection_name=collection_name
        )
        span.set_attribute("qdrant.deleted", True)
        print(
            f"Collection '{collection_name}' deleted."
        )


def delete_all_rag_collections() -> list[str]:
    with tracer.start_as_current_span(
        "qdrant.delete_rag_collections",
        attributes=span_kind(
            OpenInferenceSpanKindValues.TOOL
        )
    ) as span:

        collections = client.get_collections().collections
        deleted = []

        for collection in collections:

            if collection.name.startswith(
                "rag_documents_"
            ):

                client.delete_collection(
                    collection_name=collection.name
                )

                deleted.append(
                    collection.name
                )

                print(
                    f"Deleted: {collection.name}"
                )

        set_output(
            span,
            {
                "deleted_collections": deleted
            }
        )

        return deleted
    