from pydantic import BaseModel
from typing import Optional

class ChunkMetadata(BaseModel):
    source_file: str
    chunk_id: int
    chunk_strategy: str
    page_number: Optional[int] = None