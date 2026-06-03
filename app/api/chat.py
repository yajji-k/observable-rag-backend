from fastapi import APIRouter
from pydantic import BaseModel

from app.rag.retrieve_pipeline import run_rag


# Router for chat-related APIs
chat_router = APIRouter()


# Request body schema for chat endpoint
class ChatRequest(BaseModel):
    query: str


@chat_router.post("/chat")
def chat(request: ChatRequest):

    chunk_strategy: str = "recursive"
    
    # Run complete RAG pipeline
    response = run_rag(
        request.query,
        chunk_strategy
    )

    return {
        "response": response
    }