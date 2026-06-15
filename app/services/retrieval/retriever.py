import time

from openinference.semconv.trace import OpenInferenceSpanKindValues

from app.services.ingestion.embedder import (
    generate_embedding
)

from app.infrastructure.vector_store.qdrant import (
    client,
    query_points,
)
from app.observability.retrieval_analytics import (
    log_retrieval_score_analytics,
)
from app.observability.tracing import (
    set_attributes,
    set_input,
    set_output,
    set_retrieval_documents,
    span_kind,
    tracer,
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
    top_k: int = 3,
    telemetry_attributes: dict | None = None
) -> list:
    collection_name = (
        StrategyRegistry.get_collection_name(
            strategy=strategy
        )
    )

    with tracer.start_as_current_span(
        f"retrieval.{strategy}",
        attributes=span_kind(
            OpenInferenceSpanKindValues.RETRIEVER
        )
    ) as span:
        started_at = time.perf_counter()
        set_input(span, query, mime_type="text/plain")
        set_attributes(
            span,
            {
                "retrieval.strategy": strategy,
                "retrieval.collection": collection_name,
                "retrieval.top_k": top_k,
                **(telemetry_attributes or {}),
            }
        )

        if not collection_exists(
            collection_name
        ):
            span.set_attribute(
                "retrieval.status",
                "collection_not_found"
            )
            raise ValueError(
                f"Collection '{collection_name}' does not exist"
            )

        query_embedding = generate_embedding(query)
        results = query_points(
            collection_name=collection_name,
            query_embedding=query_embedding,
            top_k=top_k
        )

        scores = [
            float(result.score)
            for result in results
        ]
        analytics = log_retrieval_score_analytics(
            span,
            scores
        )
        set_retrieval_documents(span, results)
        set_attributes(
            span,
            {
                "retrieval.status": "success",
                "retrieval.result_count": len(results),
                "retrieval.latency_ms": round(
                    (time.perf_counter() - started_at) * 1000,
                    2
                ),
            }
        )
        set_output(
            span,
            {
                "strategy": strategy,
                "collection": collection_name,
                "result_count": len(results),
                **analytics,
            }
        )

        return results
