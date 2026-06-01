from phoenix.otel import register


# Register OpenTelemetry tracing with Phoenix
tracer_provider = register(

    project_name="observable-rag-backend",

    # Automatically instrument supported libraries
    auto_instrument=True
)