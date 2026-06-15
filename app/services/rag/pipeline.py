from openinference.instrumentation import using_attributes
from openinference.semconv.trace import OpenInferenceSpanKindValues

from app.observability.tracing import (
    set_attributes,
    set_input,
    set_output,
    span_kind,
    tracer,
)
from app.services.generation.gemini_client import generate_response
from app.services.generation.prompt_builder import build_rag_prompt
from app.services.retrieval.retriever import retrieve


def run_rag(
    query: str,
    chunk_strategy: str = "character"
):
    with using_attributes(
        metadata={
            "retrieval_strategy": chunk_strategy
        },
        tags=["rag", "chat"]
    ):
        with tracer.start_as_current_span(
            "rag.query",
            attributes=span_kind(
                OpenInferenceSpanKindValues.CHAIN
            )
        ) as span:
            set_input(span, query, mime_type="text/plain")
            set_attributes(
                span,
                {
                    "rag.retrieval_strategy": chunk_strategy,
                }
            )

            results = retrieve(
                query=query,
                strategy=chunk_strategy
            )

            context = "\n\n".join(
                result.payload["text"]
                for result in results
            )

            with tracer.start_as_current_span(
                "rag.build_prompt",
                attributes=span_kind(
                    OpenInferenceSpanKindValues.PROMPT
                )
            ) as prompt_span:
                prompt = build_rag_prompt(
                    query=query,
                    context=context
                )
                set_attributes(
                    prompt_span,
                    {
                        "prompt.context_count": len(results),
                        "prompt.context_character_count":
                            len(context),
                    }
                )
                set_output(
                    prompt_span,
                    prompt,
                    mime_type="text/plain"
                )

            response = generate_response(prompt)

            set_attributes(
                span,
                {
                    "rag.retrieved_chunk_count": len(results),
                    "rag.response_character_count":
                        len(response or ""),
                }
            )
            set_output(
                span,
                response or "",
                mime_type="text/plain"
            )

            return response
