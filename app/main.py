from fastapi import FastAPI

# Looks unused but initializes
# OpenTelemetry + Phoenix instrumentation
import app.core.telemetry

from app.api.routes.benchmark import benchmark_router
from app.api.routes.chat import chat_router
from app.api.routes.chunking import chunking_router
from app.api.routes.ingestion import ingest_router
from app.api.routes.retrieval_evaluation import eval_router

app = FastAPI()


app.include_router(chat_router)
app.include_router(ingest_router)
app.include_router(chunking_router)
app.include_router(eval_router)
app.include_router(benchmark_router)
