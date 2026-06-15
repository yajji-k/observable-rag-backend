from pathlib import Path

from openinference.semconv.trace import OpenInferenceSpanKindValues

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
from app.observability.tracing import (
    set_attributes,
    set_input,
    set_output,
    span_kind,
    tracer,
)


def run_ingestion_pipeline(
    file_path: str,
    chunk_strategy: str = "character"
):
    source_file = Path(file_path).name

    with tracer.start_as_current_span(
        "ingestion.run",
        attributes=span_kind(
            OpenInferenceSpanKindValues.CHAIN
        )
    ) as span:
        set_input(
            span,
            {
                "source_file": source_file,
                "chunk_strategy": chunk_strategy,
            }
        )
        set_attributes(
            span,
            {
                "ingestion.source_file": source_file,
                "ingestion.chunk_strategy": chunk_strategy,
            }
        )

        with tracer.start_as_current_span(
            "ingestion.load_document",
            attributes=span_kind(
                OpenInferenceSpanKindValues.TOOL
            )
        ) as load_span:
            unfiltered_text = load_pdf(file_path)
            set_attributes(
                load_span,
                {
                    "document.source_file": source_file,
                    "document.character_count":
                        len(unfiltered_text),
                }
            )

        with tracer.start_as_current_span(
            "ingestion.clean_document",
            attributes=span_kind(
                OpenInferenceSpanKindValues.TOOL
            )
        ) as clean_span:
            text = clean_text(unfiltered_text)
            set_attributes(
                clean_span,
                {
                    "document.input_character_count":
                        len(unfiltered_text),
                    "document.output_character_count":
                        len(text),
                }
            )

        try:
            chunker = ChunkerFactory.create(
                strategy=chunk_strategy,
                chunk_size=500,
                overlap=100
            )
        except ValueError as error:
            span.set_attribute(
                "ingestion.status",
                "invalid_strategy"
            )
            return {
                "status": "error",
                "message": str(error)
            }

        with tracer.start_as_current_span(
            "ingestion.chunk_document",
            attributes=span_kind(
                OpenInferenceSpanKindValues.CHAIN
            )
        ) as chunk_span:
            chunks = chunker.chunk(text)
            set_attributes(
                chunk_span,
                {
                    "chunk.strategy": chunk_strategy,
                    "chunk.count": len(chunks),
                    "chunk.size": 500,
                    "chunk.overlap": 100,
                }
            )

        chunk_records = [
            {
                "text": chunk,
                "chunk_id": index,
                "chunk_strategy": chunk_strategy,
                "source_file": source_file,
            }
            for index, chunk in enumerate(chunks)
        ]

        with tracer.start_as_current_span(
            "ingestion.embed_chunks",
            attributes=span_kind(
                OpenInferenceSpanKindValues.CHAIN
            )
        ) as embedding_span:
            embeddings = [
                generate_embedding(record["text"])
                for record in chunk_records
            ]
            embedding_span.set_attribute(
                "embedding.count",
                len(embeddings)
            )

        collection_name = f"rag_documents_{chunk_strategy}"
        create_collection(collection_name)
        insert_documents(
            collection_name,
            chunk_records,
            embeddings
        )

        response = {
            "message": "Document ingested successfully"
        }
        set_attributes(
            span,
            {
                "ingestion.status": "success",
                "ingestion.chunk_count": len(chunks),
                "ingestion.collection": collection_name,
            }
        )
        set_output(span, response)

        return response
