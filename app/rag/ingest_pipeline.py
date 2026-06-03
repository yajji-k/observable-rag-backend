from app.ingestion.loader import load_pdf

from app.ingestion.chunker import chunk_text_character, chunk_text_recursive

from app.ingestion.cleaner import clean_text

from app.ingestion.embedder import (
    generate_embedding
)

from app.retrieval.qdrant_client import (
    create_collection,
    insert_documents
)


def run_ingestion_pipeline(
    file_path: str,
    chunk_strategy: str = "character"
):

    # Extract raw text from uploaded document
    unfiltered_text = load_pdf(file_path)

    # Clean the raw text to remove noise
    text = clean_text(unfiltered_text)

    # Split large text into smaller chunks
    if chunk_strategy == "character":    
        chunks = chunk_text_character(text)
    elif chunk_strategy == "recursive":
        chunks = chunk_text_recursive(text)

    # Generate embeddings for each chunk
    embeddings = [
        generate_embedding(chunk)
        for chunk in chunks
    ]

    # Recreate Qdrant collection for fresh ingestion
    collection_name = f"rag_documents_{chunk_strategy}"  # chunking strat for collection_name dynamicall
    create_collection(collection_name)

    # Store chunks and embeddings in vector database
    insert_documents(
        collection_name,
        chunks,
        embeddings
    )

    return {
        "message": "Document ingested successfully"
    }