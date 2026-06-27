import time

from openinference.semconv.trace import OpenInferenceSpanKindValues

from app.core.config import (
    RERANKER_CANDIDATE_COUNT,
    RERANKER_ENABLED,
    RERANKER_FINAL_TOP_K,
    RERANKER_MODEL,
)
from app.services.ingestion.embedder import (
    generate_embedding
)

from app.infrastructure.vector_store.qdrant import (
    client,
    query_points,
)
from app.observability.retrieval_analytics import (
    calculate_average_score,
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
from app.services.reranking.models import RerankerConfig
from app.services.reranking.service import RerankingService


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
    telemetry_attributes: dict | None = None,
    reranking_enabled: bool | None = None,
    reranker_model: str | None = None,
    candidate_count: int | None = None
) -> list:
    collection_name = (
        StrategyRegistry.get_collection_name(
            strategy=strategy
        )
    )
    reranker_config = _build_reranker_config(
        enabled=reranking_enabled,
        model=reranker_model,
        candidate_count=candidate_count,
        final_top_k=top_k
    )
    reranking_is_enabled = reranker_config.enabled
    vector_top_k = (
        max(
            reranker_config.candidate_count,
            top_k
        )
        if reranking_is_enabled
        else top_k
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
                "retrieval.vector_top_k": vector_top_k,
                "retrieval.reranking_enabled":
                    reranking_is_enabled,
                "retrieval.reranker_model":
                    reranker_config.model
                    if reranking_is_enabled
                    else None,
                "retrieval.candidate_count":
                    reranker_config.candidate_count
                    if reranking_is_enabled
                    else top_k,
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

        with tracer.start_as_current_span(
            "retrieval.vector_search",
            attributes=span_kind(
                OpenInferenceSpanKindValues.RETRIEVER
            )
        ) as vector_span:
            vector_started_at = time.perf_counter()
            set_attributes(
                vector_span,
                {
                    "retrieval.collection": collection_name,
                    "retrieval.top_k": vector_top_k,
                }
            )
            results = query_points(
                collection_name=collection_name,
                query_embedding=query_embedding,
                top_k=vector_top_k
            )
            vector_search_time_ms = round(
                (
                    time.perf_counter()
                    - vector_started_at
                ) * 1000,
                2
            )
            set_attributes(
                vector_span,
                {
                    "retrieval.result_count":
                        len(results),
                    "retrieval.vector_search_time_ms":
                        vector_search_time_ms,
                }
            )

        scores = [
            float(result.score)
            for result in results
        ]
        _attach_vector_scores(results)

        rerank_time_ms = 0.0
        rerank_scores: list[float] = []
        average_rerank_score = None

        if reranking_is_enabled:
            with tracer.start_as_current_span(
                "retrieval.reranking",
                attributes=span_kind(
                    OpenInferenceSpanKindValues.RERANKER
                )
            ) as rerank_span:
                reranking_service = RerankingService(
                    reranker_config
                )
                reranking_result = reranking_service.rerank(
                    query=query,
                    chunks=results,
                    top_k=top_k
                )
                results = reranking_result.chunks
                rerank_scores = reranking_result.scores
                rerank_time_ms = reranking_result.elapsed_ms
                average_rerank_score = (
                    reranking_result.average_score
                )

                print("Model:", reranking_service.model_name)
                print("Scores:", rerank_scores)
                print("Average:", average_rerank_score)
                print("Time:", rerank_time_ms)

                set_attributes(
                    rerank_span,
                    {
                        "retrieval.reranker_model":
                            reranking_service.model_name,
                        "retrieval.candidate_count":
                            len(scores),
                        "retrieval.returned_count":
                            len(results),
                        "retrieval.rerank_scores":
                            rerank_scores,
                        "retrieval.average_rerank_score":
                            average_rerank_score,
                        "retrieval.rerank_time_ms":
                            rerank_time_ms,
                    }
                )
                set_output(
                    rerank_span,
                    {
                        "returned_count": len(results),
                        "rerank_scores": rerank_scores,
                        "average_rerank_score":
                            average_rerank_score,
                    }
                )
        else:
            results = results[:top_k]

        total_retrieval_time_ms = round(
            (time.perf_counter() - started_at) * 1000,
            2
        )
        average_vector_score = calculate_average_score(
            scores
        )
        analytics = log_retrieval_score_analytics(
            span,
            scores
        )
        reranked_chunk_order = [
            (result.payload or {}).get("chunk_id")
            for result in results
        ]
        set_retrieval_documents(span, results)
        set_attributes(
            span,
            {
                "retrieval.status": "success",
                "retrieval.result_count": len(results),
                "retrieval.latency_ms":
                    total_retrieval_time_ms,
                "retrieval.vector_search_time_ms":
                    vector_search_time_ms,
                "retrieval.rerank_time_ms":
                    rerank_time_ms,
                "retrieval.total_retrieval_time_ms":
                    total_retrieval_time_ms,
                "retrieval.average_vector_score":
                    average_vector_score,
                "retrieval.average_rerank_score":
                    average_rerank_score,
                "retrieval.reranker_model":
                    reranker_config.model
                    if reranking_is_enabled
                    else None,
                "retrieval.candidate_count":
                    len(scores),
                "retrieval.returned_count":
                    len(results),
                "retrieval.reranked_chunk_order":
                    reranked_chunk_order,
            }
        )
        set_output(
            span,
            {
                "strategy": strategy,
                "collection": collection_name,
                "result_count": len(results),
                "vector_search_time_ms":
                    vector_search_time_ms,
                "rerank_time_ms": rerank_time_ms,
                "total_retrieval_time_ms":
                    total_retrieval_time_ms,
                "average_vector_score":
                    average_vector_score,
                "average_rerank_score":
                    average_rerank_score,
                "reranker_model":
                    reranker_config.model
                    if reranking_is_enabled
                    else None,
                "candidate_count": len(scores),
                "returned_count": len(results),
                "reranked_chunk_order":
                    reranked_chunk_order,
                "reranking_enabled":
                    reranking_is_enabled,
                **analytics,
            }
        )

        return results


def _build_reranker_config(
    enabled: bool | None,
    model: str | None,
    candidate_count: int | None,
    final_top_k: int | None
) -> RerankerConfig:
    return RerankerConfig(
        enabled=(
            RERANKER_ENABLED
            if enabled is None
            else enabled
        ),
        model=model or RERANKER_MODEL,
        candidate_count=(
            candidate_count
            or RERANKER_CANDIDATE_COUNT
        ),
        final_top_k=(
            final_top_k
            or RERANKER_FINAL_TOP_K
        )
    )


def _attach_vector_scores(results: list) -> None:
    for result in results:
        payload = result.payload or {}
        payload["vector_score"] = float(result.score)
        payload.setdefault("rerank_score", None)
        result.payload = payload
