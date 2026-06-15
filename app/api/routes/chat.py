from fastapi import APIRouter

from app.schemas.chat import ChatRequest
from app.services.rag.pipeline import run_rag


# Router for chat-related APIs
chat_router = APIRouter()


@chat_router.post("/chat")
def chat(request: ChatRequest):
    
    # Run complete RAG pipeline
    response = run_rag(
        request.query,
        request.chunk_strat
    )

    return {
        "response": response
    }
