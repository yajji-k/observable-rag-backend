from app.ingestion.loader import (
    load_pdf
)

from app.ingestion.chunker import (
    chunk_text
)

from app.ingestion.embedder import (
    generate_embedding
)

from app.retrieval.qdrant_client import (
    create_collection,
    insert_documents
)


# Load PDF content
text = load_pdf(
    "data/TA_wrkbk.pdf"
)


# Split document into chunks
chunks = chunk_text(text)


# Generate embeddings for each chunk
embeddings = [
    generate_embedding(chunk)
    for chunk in chunks
]


# Recreate vector collection
create_collection()


# Store embeddings in Qdrant
insert_documents(
    chunks,
    embeddings
)