import os

from dotenv import load_dotenv


# Load environment variables from .env file
load_dotenv()


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
    "COLLECTION_NAME",
    "rag_documents"
)


# Gemini API key
GEMINI_API_KEY = os.getenv(
    "GEMINI_API_KEY"
)