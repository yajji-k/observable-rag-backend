from pydantic import BaseModel

# Request body schema for chat endpoint
class ChatRequest(BaseModel):
    query: str