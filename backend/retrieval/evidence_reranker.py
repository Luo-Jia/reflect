
from datetime import datetime


# Provisional study-design priority weights.
# These are NOT validated methodological quality scores.

EVIDENCE_WEIGHTS = {
    "meta-analysis": 1.00,
    "systematic review": 0.95,
    "randomized controlled trial": 0.90,
    "clinical trial": 0.75,
    "controlled clinical trial": 0.80,
    "cohort study": 0.60,
    "observational study": 0.50,
    "case-control study": 0.50,
    "cross-sectional study": 0.40,
    "case reports": 0.20,
    "review": 0.35,
}


def classify_evidence(publication_types):

    types = [
        item.lower()
        for item in (publication_types or [])
    ]

    # Prefer more specific publication types.
    if "meta-analysis" in types:
        return "Meta-analysis", 1.00

    if "systematic review" in types:
        return "Systematic review", 0.95

    if "randomized controlled trial" in types:
        return "Randomized controlled trial", 0.90

    if "controlled clinical trial" in types:
        return "Controlled clinical trial", 0.80

    if "clinical trial" in types:
        return "Clinical trial", 0.75

    if "cohort study" in types:
        return "Cohort study", 0.60

    if "case-control study" in types:
        return "Case-control study", 0.50

    if "observational study" in types:
        return "Observational study", 0.50

    if "cross-sectional study" in types:
        return "Cross-sectional study", 0.40

    if "case reports" in types:
        return "Case report", 0.20

    if "review" in types:
        return "Review", 0.35

    return "Unclassified", 0.30


def rerank_by_evidence(
    articles,
    relevance_weight=0.60,
    evidence_weight=0.40,
):
    """
    Rerank retrieved articles using relevance and
    provisional study-design priority.

    This does not assess risk of bias or certainty of evidence.
    """

    if not articles:
        return []

    if abs(relevance_weight + evidence_weight - 1.0) > 1e-6:
        raise ValueError("Weights must sum to 1.0")

    reranked = []

    for article in articles:

        # MMR score is used as the retrieval relevance signal.
        mmr_score = float(
            article.get("mmr_score", 0.0)
        )

        # Map [-1, 1] approximately into [0, 1].
        relevance = max(
            0.0,
            min(1.0, (mmr_score + 1.0) / 2.0)
        )

        evidence_type, design_score = classify_evidence(
            article.get("publication_types", [])
        )

        final_score = (
            relevance_weight * relevance
            + evidence_weight * design_score
        )

        ranked_article = article.copy()

        ranked_article["evidence_type"] = evidence_type
        ranked_article["design_priority"] = design_score
        ranked_article["relevance_score"] = relevance
        ranked_article["evidence_score"] = round(
            final_score, 4
        )

        reranked.append(ranked_article)

    reranked.sort(
        key=lambda x: x["evidence_score"],
        reverse=True,
    )

    return reranked
