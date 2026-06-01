def chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 100
):

    # Store generated text chunks
    chunks = []

    # Starting index for chunk slicing
    start = 0

    while start < len(text):

        # Define chunk boundary
        end = start + chunk_size

        # Extract chunk from text
        chunk = text[start:end]

        chunks.append(chunk)

        # Move forward while keeping overlap
        start += chunk_size - overlap

    return chunks