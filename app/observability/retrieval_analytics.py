from typing import Any

from opentelemetry.trace import Span


def calculate_retrieval_score_analytics(
    scores: list[float]
) -> dict[str, Any]:
    if not scores:
        return {
            "retrieval.max_score": None,
            "retrieval.min_score": None,
            "retrieval.average_score": None,
            "retrieval.score_range": None,
        }

    return {
        "retrieval.max_score": round(max(scores), 4),
        "retrieval.min_score": round(min(scores), 4),
        "retrieval.average_score": round(
            sum(scores) / len(scores),
            4
        ),
        "retrieval.score_range": round(
            max(scores) - min(scores),
            4
        ),
    }


def log_retrieval_score_analytics(
    span: Span,
    scores: list[float]
) -> dict[str, Any]:
    analytics = calculate_retrieval_score_analytics(
        scores
    )

    for name, value in analytics.items():
        if value is not None:
            span.set_attribute(name, value)

    return analytics


def calculate_average_score(
    scores: list[float]
) -> float | None:
    if not scores:
        return None

    return round(
        sum(scores) / len(scores),
        4
    )
