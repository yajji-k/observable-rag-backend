from app.services.ingestion.loader import load_pdf

from app.services.ingestion.chunking.chunker_factory import ChunkerFactory

from app.services.ingestion.cleaner import clean_text

from app.services.ingestion.embedder import (
    generate_embedding
)

from app.infrastructure.vector_store.qdrant import (
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

    try:
        # Create chunking strategy
        chunker = ChunkerFactory.create(
            strategy=chunk_strategy,
            chunk_size=500,
            overlap=100
        )
    
    except ValueError as e:
        return {
            "status" : "error",
            "message" : str(e)
        }

    # Generate chunks
    chunks = chunker.chunk(text)

    # Store chunk metadata
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

    # Generate embeddings
    embeddings = [
        generate_embedding(record["text"])
        for record in chunk_records
    ]

    # Create collection
    collection_name = f"rag_documents_{chunk_strategy}"

    create_collection(collection_name)

    # Insert into Qdrant
    insert_documents(
        collection_name,
        chunk_records,
        embeddings
    )

    return {
        "message": "Document ingested successfully"
    }
