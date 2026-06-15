from app.services.generation.gemini_client import (
    generate_response
)

from app.services.generation.prompt_builder import (
    build_rag_prompt
)

from app.services.retrieval.retriever import (
    retrieve
)


# Sample user query
query = "Explain technical analysis in simple words."


# Retrieve relevant chunks
results = retrieve(
    query=query,
    strategy="character"
)


# Build context from retrieved chunks
context = "\n\n".join(
    [
        result.payload["text"]
        for result in results
    ]
)


# Generate final RAG prompt
prompt = build_rag_prompt(
    query=query,
    context=context
)


# Generate LLM response
response = generate_response(
    prompt
)

print(response)
