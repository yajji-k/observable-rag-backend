from app.services.generation.gemini_client import generate_response
from app.services.generation.prompt_builder import build_rag_prompt
from app.services.retrieval.retriever import retrieve


def run_rag(
    query: str,
    chunk_strategy: str = "character"
):
    results = retrieve(
        query=query,
        strategy=chunk_strategy
    )

    context = "\n\n".join(
        result.payload["text"]
        for result in results
    )

    prompt = build_rag_prompt(
        query=query,
        context=context
    )

    response = generate_response(
        prompt
    )

    return response
