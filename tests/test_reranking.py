from qdrant_client.models import ScoredPoint

from app.services.reranking.bge_reranker import BGEReranker


class FakeCrossEncoder:

    def predict(self, pairs):
        return [0.1, 0.9, 0.4]


def test_bge_reranker_preserves_vector_score_and_sorts_by_rerank_score():
    reranker = BGEReranker.__new__(BGEReranker)
    reranker._model_name = "fake-bge"
    reranker._model = FakeCrossEncoder()

    chunks = [
        ScoredPoint(
            id="1",
            version=1,
            score=0.8,
            payload={
                "text": "first",
                "chunk_id": 1,
            }
        ),
        ScoredPoint(
            id="2",
            version=1,
            score=0.7,
            payload={
                "text": "second",
                "chunk_id": 2,
            }
        ),
        ScoredPoint(
            id="3",
            version=1,
            score=0.6,
            payload={
                "text": "third",
                "chunk_id": 3,
            }
        ),
    ]

    reranked = reranker.rerank(
        query="query",
        chunks=chunks,
        top_k=2
    )

    assert [
        chunk.payload["chunk_id"]
        for chunk in reranked
    ] == [2, 3]

    assert reranked[0].score == 0.7
    assert reranked[0].payload["vector_score"] == 0.7
    assert reranked[0].payload["rerank_score"] == 0.9
