from fastapi import FastAPI

# Looks unused but initializes
# OpenTelemetry + Phoenix instrumentation
import app.telemetry

from app.api.chat import chat_router

from app.api.ingest import ingest_router

from app.api.chunking import chunking_router

from app.evaluation.api.evaluation_api import eval_router

from app.evaluation.api.benchmark_api import benchmark_router

# Initialize FastAPI application
app = FastAPI()


# Register API routes
app.include_router(chat_router)
app.include_router(ingest_router)
app.include_router(chunking_router)
app.include_router(eval_router)
app.include_router(benchmark_router)