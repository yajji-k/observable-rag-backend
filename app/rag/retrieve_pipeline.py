from opentelemetry import trace

from app.retrieval.retriever import search

from app.generation.gemini_client import (
    generate_response
)

from app.generation.prompt_builder import (
    build_rag_prompt
)


# OpenTelemetry tracer for RAG pipeline tracing
tracer = trace.get_tracer(__name__)


def run_rag(query: str):

    # Retrieve relevant chunks from vector database
    with tracer.start_as_current_span(
        "vector_retrieval"
    ) as span:

        span.set_attribute(
            "user.query",
            query
        )

        results = search(query)

    # Construct prompt using retrieved context
    with tracer.start_as_current_span(
        "prompt_construction"
    ) as span:

        context = "\n\n".join(
            [
                result.payload["text"]
                for result in results
            ]
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

    # Generate final response using Gemini
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