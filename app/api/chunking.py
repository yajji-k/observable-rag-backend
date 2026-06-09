from fastapi import ( APIRouter, Form, UploadFile, File )

from app.ingestion.chunking.chunker_factory import ChunkerFactory

chunking_router = APIRouter

@chunking_router.get("/chunking/strategies")
def get_chunking_strategies():
    return {
        "strategies": ChunkerFactory.get_available_strategies()
    }