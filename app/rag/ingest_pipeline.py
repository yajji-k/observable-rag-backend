from app.ingestion.loader import load_pdf

from app.ingestion.chunker import chunk_text

from app.ingestion.embedder import (
    generate_embedding
)

from app.retrieval.qdrant_client import (
    create_collection,
    insert_documents
)


def run_ingestion_pipeline(
    file_path: str
):

    # Extract raw text from uploaded document
    text = load_pdf(file_path)

    # Split large text into smaller chunks
    chunks = chunk_text(text)

    # Generate embeddings for each chunk
    embeddings = [
        generate_embedding(chunk)
        for chunk in chunks
    ]

    # Recreate Qdrant collection for fresh ingestion
    create_collection()

    # Store chunks and embeddings in vector database
    insert_documents(
        chunks,
        embeddings
    )

    return {
        "message": "Document ingested successfully"
    }