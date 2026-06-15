import json
from collections.abc import Mapping, Sequence
from typing import Any

from openinference.semconv.trace import (
    DocumentAttributes,
    OpenInferenceSpanKindValues,
    SpanAttributes,
)
from opentelemetry import trace
from opentelemetry.trace import Span

from app.core.config import PHOENIX_CAPTURE_CONTENT


tracer = trace.get_tracer("observable-rag-system")
JSON_MIME_TYPE = "application/json"
TEXT_MIME_TYPE = "text/plain"
MAX_CONTENT_LENGTH = 8000


def span_kind(kind: OpenInferenceSpanKindValues) -> dict[str, str]:
    return {
        SpanAttributes.OPENINFERENCE_SPAN_KIND: kind.value
    }


def set_attributes(
    span: Span,
    attributes: Mapping[str, Any]
) -> None:
    for key, value in attributes.items():
        if value is not None:
            span.set_attribute(key, value)


def set_input(
    span: Span,
    value: Any,
    mime_type: str = JSON_MIME_TYPE
) -> None:
    if not PHOENIX_CAPTURE_CONTENT:
        return

    span.set_attribute(
        SpanAttributes.INPUT_VALUE,
        _serialize(value)
    )
    span.set_attribute(
        SpanAttributes.INPUT_MIME_TYPE,
        mime_type
    )


def set_output(
    span: Span,
    value: Any,
    mime_type: str = JSON_MIME_TYPE
) -> None:
    if not PHOENIX_CAPTURE_CONTENT:
        return

    span.set_attribute(
        SpanAttributes.OUTPUT_VALUE,
        _serialize(value)
    )
    span.set_attribute(
        SpanAttributes.OUTPUT_MIME_TYPE,
        mime_type
    )


def set_retrieval_documents(
    span: Span,
    results: Sequence[Any]
) -> None:
    for index, result in enumerate(results):
        payload = result.payload or {}
        prefix = f"{SpanAttributes.RETRIEVAL_DOCUMENTS}.{index}"

        if PHOENIX_CAPTURE_CONTENT:
            content = str(payload.get("text", ""))
            span.set_attribute(
                f"{prefix}.{DocumentAttributes.DOCUMENT_CONTENT}",
                _truncate(content)
            )

        document_id = payload.get("chunk_id")
        if document_id is not None:
            span.set_attribute(
                f"{prefix}.{DocumentAttributes.DOCUMENT_ID}",
                str(document_id)
            )

        span.set_attribute(
            f"{prefix}.{DocumentAttributes.DOCUMENT_SCORE}",
            float(result.score)
        )

        metadata = {
            key: value
            for key, value in payload.items()
            if key != "text"
        }
        span.set_attribute(
            f"{prefix}.{DocumentAttributes.DOCUMENT_METADATA}",
            json.dumps(metadata, default=str)
        )


def _serialize(value: Any) -> str:
    if isinstance(value, str):
        return _truncate(value)

    return _truncate(
        json.dumps(value, default=str)
    )


def _truncate(value: str) -> str:
    return value[:MAX_CONTENT_LENGTH]
