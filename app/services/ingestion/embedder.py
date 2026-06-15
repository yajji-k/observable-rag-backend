from functools import lru_cache

from openinference.semconv.trace import (
    OpenInferenceSpanKindValues,
    SpanAttributes,
)
from sentence_transformers import SentenceTransformer

from app.observability.tracing import (
    set_attributes,
    set_input,
    set_output,
    span_kind,
    tracer,
)


EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    return SentenceTransformer(
        EMBEDDING_MODEL_NAME
    )


def generate_embedding(text: str):
    with tracer.start_as_current_span(
        "embedding.generate",
        attributes=span_kind(
            OpenInferenceSpanKindValues.EMBEDDING
        )
    ) as span:
        set_input(span, text, mime_type="text/plain")
        set_attributes(
            span,
            {
                SpanAttributes.EMBEDDING_MODEL_NAME:
                    EMBEDDING_MODEL_NAME,
                "embedding.input_length": len(text),
            }
        )

        embedding = get_embedding_model().encode(text)
        vector = embedding.tolist()

        set_output(
            span,
            {
                "dimensions": len(vector)
            }
        )

        return vector
