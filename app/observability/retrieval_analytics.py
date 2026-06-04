def log_retrieval_score_analytics(
    span,
    scores
):
    """
    Calculate retrieval score metrics and
    attach them to the OpenTelemetry span.
    """

    

    if not scores:
        return

    span.set_attribute(
        "retrieval.max_score",
        round(max(scores), 4)
    )

    span.set_attribute(
        "retrieval.min_score",
        round(min(scores), 4)
    )

    span.set_attribute(
        "retrieval.avg_score",
        round(
            sum(scores) / len(scores),
            4
        )
    )
    
    span.set_attribute(
    "retrieval.score_range",
    round(
        max(scores) - min(scores),
        4
    )
)