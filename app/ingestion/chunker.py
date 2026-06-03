from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_text_character(
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


def chunk_text_recursive(
    text: str,
    chunk_size: int = 500,
    overlap: int = 100    
):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ]
    )
    
    chunks = splitter.split_text(text)
    
    return chunks

