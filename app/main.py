from fastapi import FastAPI

# Looks unused but initializes
# OpenTelemetry + Phoenix instrumentation
import app.telemetry

from app.api.chat import chat_router

from app.api.ingest import ingest_router

from app.api.chunking import chunking_router

# Initialize FastAPI application
app = FastAPI()


# Register API routes
app.include_router(chat_router)
app.include_router(ingest_router)
app.include_router(chunking_router)