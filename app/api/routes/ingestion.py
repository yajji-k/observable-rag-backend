from fastapi import (
    APIRouter,
    Form,
    UploadFile,
    File
)
from typing import Annotated

import os

from app.core.config import INGESTION_FOLDER
from app.services.ingestion.pipeline import (
    run_ingestion_pipeline
)

from app.infrastructure.vector_store.qdrant import (
    delete_all_rag_collections
)

from pathlib import Path


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

@ingest_router.post("/ingest/folder")
def ingest_folder(
    clear_existing: bool = False,
    chunk_strategy: str = "character"
):

    if clear_existing:
        delete_all_rag_collections()

    ingestion_dir = Path(INGESTION_FOLDER)

    if not ingestion_dir.exists():
        return {
            "status": "error",
            "message": "Ingestion folder not found"
        }

    pdf_files = list(
        ingestion_dir.glob("*.pdf")
    )

    if not pdf_files:
        return {
            "status": "error",
            "message": "No PDF files found in ingestion folder"
        }

    results = []
    failures = []

    for pdf_file in pdf_files:

        try:

            response = run_ingestion_pipeline(
                str(pdf_file),
                chunk_strategy
            )

            results.append({
                "file": pdf_file.name,
                "status": "success",
                "result": response
            })

        except Exception as e:

            failures.append({
                "file": pdf_file.name,
                "status": "failed",
                "error": str(e)
            })

    return {
        "status": "completed",
        "chunk_strategy": chunk_strategy,
        "processed_files": len(results),
        "failed_files": len(failures),
        "results": results,
        "failures": failures
    }
    
    
@ingest_router.delete("/collections/delete")
def clear_all_collections():

    deleted = delete_all_rag_collections()

    return {
        "status": "success",
        "deleted_collections": deleted
    }