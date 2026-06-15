import os

from phoenix.otel import register

from app.core.config import (
    PHOENIX_API_KEY,
    PHOENIX_BATCH_EXPORT,
    PHOENIX_COLLECTOR_ENDPOINT,
    PHOENIX_CAPTURE_CONTENT,
    PHOENIX_ENABLED,
    PHOENIX_PROJECT_NAME,
    PHOENIX_PROTOCOL,
)


tracer_provider = None


def initialize_telemetry():
    
    print("PHOENIX_COLLECTOR_ENDPOINT =", PHOENIX_COLLECTOR_ENDPOINT)
    print("PHOENIX_PROTOCOL =", PHOENIX_PROTOCOL)
    
    global tracer_provider

    if not PHOENIX_ENABLED or tracer_provider is not None:
        return tracer_provider

    if not PHOENIX_CAPTURE_CONTENT:
        os.environ["OPENINFERENCE_HIDE_INPUTS"] = "true"
        os.environ["OPENINFERENCE_HIDE_OUTPUTS"] = "true"

    tracer_provider = register(
        endpoint=PHOENIX_COLLECTOR_ENDPOINT,
        project_name=PHOENIX_PROJECT_NAME,
        batch=PHOENIX_BATCH_EXPORT,
        auto_instrument=True,
        api_key=PHOENIX_API_KEY,
        protocol=PHOENIX_PROTOCOL,
        verbose=False,
    )

    return tracer_provider


def shutdown_telemetry() -> None:
    if tracer_provider is not None:
        tracer_provider.shutdown()
