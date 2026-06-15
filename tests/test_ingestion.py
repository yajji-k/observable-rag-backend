from app.services.ingestion.loader import (
    load_pdf
)

from app.services.ingestion.chunking.chunker_factory import (
    ChunkerFactory
)

from app.services.ingestion.embedder import (
    generate_embedding
)

from app.infrastructure.vector_store.qdrant import (
    create_collection,
    insert_documents
)


# Load PDF content
text = load_pdf(
    "data/TA_wrkbk.pdf"
)


# Split document into chunks
chunker = ChunkerFactory.create("character")
chunks = chunker.chunk(text)


# Generate embeddings for each chunk
embeddings = [
    generate_embedding(chunk)
    for chunk in chunks
]


# Recreate vector collection
collection_name = "rag_documents_character"
create_collection(collection_name)


# Store embeddings in Qdrant
insert_documents(
    collection_name,
    [
        {
            "text": chunk,
            "source_file": "TA_wrkbk.pdf",
            "chunk_id": index,
            "chunk_strategy": "character",
        }
        for index, chunk in enumerate(chunks)
    ],
    embeddings
)
