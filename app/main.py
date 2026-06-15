from fastapi import FastAPI, Request
from openinference.semconv.trace import (
    OpenInferenceSpanKindValues,
)
from opentelemetry import trace
from opentelemetry.trace import SpanKind

from app.core.telemetry import (
    initialize_telemetry,
    shutdown_telemetry,
)
from app.observability.tracing import span_kind
from app.api.routes.benchmark import benchmark_router
from app.api.routes.chat import chat_router
from app.api.routes.chunking import chunking_router
from app.api.routes.ingestion import ingest_router
from app.api.routes.retrieval_evaluation import eval_router


initialize_telemetry()
http_tracer = trace.get_tracer(
    "observable-rag-system.http"
)

app = FastAPI()


@app.middleware("http")
async def trace_http_request(
    request: Request,
    call_next
):
    span_name = (
        f"{request.method} {request.url.path}"
    )

    with http_tracer.start_as_current_span(
        span_name,
        kind=SpanKind.SERVER,
        attributes={
            **span_kind(
                OpenInferenceSpanKindValues.CHAIN
            ),
            "http.request.method": request.method,
            "url.path": request.url.path,
        }
    ) as span:
        response = await call_next(request)
        span.set_attribute(
            "http.response.status_code",
            response.status_code
        )

        return response


@app.on_event("shutdown")
def shutdown_observability() -> None:
    shutdown_telemetry()


app.include_router(chat_router)
app.include_router(ingest_router)
app.include_router(chunking_router)
app.include_router(eval_router)
app.include_router(benchmark_router)
