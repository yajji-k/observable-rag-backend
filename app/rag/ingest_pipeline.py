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


    # set chunks with records(metadata) in a list
    chunk_records = []
    
    for idx, chunk in enumerate(chunks):
        chunk_records.append(
            {
                "text": chunk,
                "chunk_id": idx,
                "chunk_strategy": chunk_strategy,
                "source_file": file_path.split("/")[-1]
            }
        )

    # Generate embeddings for each chunk
    embeddings = [
        generate_embedding(records['text'])
        for records in chunk_records
    ]

    # Recreate Qdrant collection for fresh ingestion
    collection_name = f"rag_documents_{chunk_strategy}"  # chunking strat for collection_name dynamicall
    create_collection(collection_name)

    # Store chunks and embeddings in vector database
    insert_documents(
        collection_name,
        chunk_records,
        embeddings
    )

    return {
        "message": "Document ingested successfully"
    }