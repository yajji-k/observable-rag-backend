from app.services.retrieval.retriever import (
    retrieve
)


# Perform semantic similarity search
results = retrieve(
    query="What is technical analysis?",
    strategy="character"
)


# Display retrieved chunks with similarity scores
for result in results:

    print("\n")

    print("Score:", result.score)

    print(result.payload["text"])
