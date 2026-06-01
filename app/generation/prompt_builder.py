def build_rag_prompt(
    query: str,
    context: str
):

    prompt = f"""
You are a helpful AI assistant.

Answer ONLY using the provided context.

If the answer is not present,
say you do not know.

Context:
{context}

User Question:
{query}
"""

    return prompt