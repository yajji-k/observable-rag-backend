from fastapi import ( APIRouter, Form, UploadFile, File )
from typing import Annotated

import os
from app.rag.ingest_pipeline import ( run_ingestion_pipeline )


# Router for ingestion-related APIs
ingest_router = APIRouter()


@ingest_router.post("/ingest")
async def ingest_pdf(
    file: UploadFile = File(...),
    chunk_strategy: Annotated[str, Form()] = "character"
):

    # Create uploads directory if it does not exist
    os.makedirs(
        "uploads",
        exist_ok=True
    )

    # Save uploaded file temporarily for processing
    file_path = f"uploads/{file.filename}"

    with open(file_path, "wb") as pdf_file:

        content = await file.read()

        pdf_file.write(content)

    # Run complete ingestion pipeline
    response = run_ingestion_pipeline(
        file_path,
        chunk_strategy
    )

    return response