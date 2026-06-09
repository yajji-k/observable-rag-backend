from opentelemetry import trace

from app.retrieval.retriever import search
from app.generation.gemini_client import generate_response
from app.generation.prompt_builder import build_rag_prompt
from app.observability.retrieval_analytics import log_retrieval_score_analytics


# Tracer used for Phoenix observability
tracer = trace.get_tracer(__name__)


def run_rag(
    query: str,
    chunk_strategy: str = "character"
):
    
    # Retrieve relevant chunks from Qdrant
    with tracer.start_as_current_span(
        "vector_retrieval"
    ) as span:

        collection_name = f"rag_documents_{chunk_strategy}"

        span.set_attribute(
            "user.query",
            query
        )

        span.set_attribute(
            "collection_name",
            collection_name
        )

        results = search(
            query,
            collection_name
        )

        scores = [
            result.score
            for result in results
        ]

        # Analyze retrieval quality
        log_retrieval_score_analytics(
            span,
            scores
        )

        # Log retrieved chunks for debugging
        retrieval_info = [
            {
                "rank": rank,
                "score": round(result.score, 4),
                "source_file": result.payload["source_file"],
                "chunk_id": result.payload["chunk_id"],
            }
            for rank, result in enumerate(
                results,
                start=1
            )
        ]

        span.set_attribute(
            "retrieval.results",
            str(retrieval_info)
        )

    # Build prompt using retrieved context
    with tracer.start_as_current_span(
        "prompt_construction"
    ) as span:

        context = "\n\n".join(
            result.payload["text"]
            for result in results
        )

        span.set_attribute(
            "rag.context",
            context
        )

        prompt = build_rag_prompt(
            query=query,
            context=context
        )

        span.set_attribute(
            "user.prompt",
            prompt
        )

    # Generate final answer from Gemini
    with tracer.start_as_current_span(
        "llm_generation"
    ) as span:

        response = generate_response(
            prompt
        )

        span.set_attribute(
            "llm.response",
            response
        )

    return response