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


def run_rag(
    query: str,
    chunk_strategy: str = "character"
):

    # Retrieve relevant chunks from vector database
    with tracer.start_as_current_span(
        "vector_retrieval"
    ) as span:

        span.set_attribute(
            "user.query",
            query
        )

        collection_name = f"rag_documents_{chunk_strategy}"
        
        span.set_attribute(
            "collection_name", 
            collection_name
        )
        
        results = search(query, collection_name)
        
        retrieval_info = []
        
        for rank, result in enumerate(results):
            retrieval_info.append(
                {
                    "rank":  rank,
                    "score": round(result.score,4),
                    "source_file": result.payload["source_file"],
                    "chunk_id": result.payload["chunk_id"],
                }
            )
        
        span.set_attribute(
            "retrieval.results",
            str(retrieval_info)
        )

    # Construct prompt using retrieved context
    with tracer.start_as_current_span(
        "prompt_construction"
    ) as span:

        context = "\n\n".join(
            [
                result.payload["text"] for result in results
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