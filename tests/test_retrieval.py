from app.retrieval.retriever import (
    search
)


# Perform semantic similarity search
results = search(
    "What is technical analysis?"
)


# Display retrieved chunks with similarity scores
for result in results:

    print("\n")

    print("Score:", result.score)

    print(result.payload["text"])