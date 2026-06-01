from app.generation.gemini_client import (
    generate_response
)

from app.generation.prompt_builder import (
    build_rag_prompt
)

from app.retrieval.retriever import (
    search
)


# Sample user query
query = "Explain technical analysis in simple words."


# Retrieve relevant chunks
results = search(query)


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