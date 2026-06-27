import os

from dotenv import load_dotenv


# Load environment variables from .env file
load_dotenv()


def get_bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)

    if value is None:
        return default

    return value.lower() in {"1", "true", "yes", "on"}


# Qdrant configuration
QDRANT_HOST = os.getenv(
    "QDRANT_HOST",
    "localhost"
)

QDRANT_PORT = int(
    os.getenv(
        "QDRANT_PORT",
        6333
    )
)


# Vector database collection name
COLLECTION_NAME = os.getenv(
    "rag_documents"
)


# Gemini API key
GEMINI_API_KEY = os.getenv(
    "GEMINI_API_KEY"
)


# Phoenix and OpenTelemetry configuration
PHOENIX_ENABLED = get_bool_env(
    "PHOENIX_ENABLED",
    True
)

PHOENIX_PROJECT_NAME = os.getenv(
    "PHOENIX_PROJECT_NAME",
    "observable-rag-backend"
)

PHOENIX_COLLECTOR_ENDPOINT = os.getenv(
    "PHOENIX_COLLECTOR_ENDPOINT",
    "http://localhost:4317"
)

PHOENIX_PROTOCOL = os.getenv(
    "PHOENIX_PROTOCOL",
    "grpc"
)

PHOENIX_API_KEY = os.getenv(
    "PHOENIX_API_KEY"
)

PHOENIX_BATCH_EXPORT = get_bool_env(
    "PHOENIX_BATCH_EXPORT",
    True
)

PHOENIX_CAPTURE_CONTENT = get_bool_env(
    "PHOENIX_CAPTURE_CONTENT",
    True
)

INGESTION_FOLDER = os.getenv(
    "INGESTION_FOLDER", 
    "data/ingestion"
)


# Reranker configuration
RERANKER_ENABLED = get_bool_env(
    "RERANKER_ENABLED",
    True
)

RERANKER_MODEL = os.getenv(
    "RERANKER_MODEL",
    "bge"
)

RERANKER_BGE_MODEL_NAME = os.getenv(
    "RERANKER_BGE_MODEL_NAME",
    "BAAI/bge-reranker-base"
)

RERANKER_CANDIDATE_COUNT = int(
    os.getenv(
        "RERANKER_CANDIDATE_COUNT",
        20
    )
)

RERANKER_FINAL_TOP_K = int(
    os.getenv(
        "RERANKER_FINAL_TOP_K",
        5
    )
)
