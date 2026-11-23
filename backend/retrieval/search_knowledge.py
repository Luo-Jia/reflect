
from functools import lru_cache

from backend.model.embedding_model import EmbeddingModel
from backend.database.vector_database import VectorDatabase
from backend.retrieval.evidence_reranker import rerank_by_evidence

import logging

logger = logging.getLogger(__spec__.name if __spec__ else __name__)


@lru_cache(maxsize=1)
def get_embedding_model():
    return EmbeddingModel()


def search_research(question, top_k=5, fetch_k=30, lambda_mult=0.7):
    logger.debug("Searching vector database")

    embedding_model = get_embedding_model()
    database = VectorDatabase()

    try:

        # Convert user question into an embedding
        query_embedding = embedding_model.generate_embedding(
            question
        )

        # # Search pgvector
        # candidates = database.search_similar(
        #     embedding=query_embedding,
        #     limit=top_k,
        # )

        # Step 2: Retrieve diverse candidates using MMR
        candidates = database.search_mmr(
            embedding=query_embedding,
            limit=top_k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
        )

        # Step 3: Rerank by evidence design and relevance
        ranked_articles = rerank_by_evidence(
            candidates,
            relevance_weight=0.60,
            evidence_weight=0.40,
        )

        # Step 4: Return top-ranked publications
        return ranked_articles[:top_k]

        # return candidates

    finally:
        database.close()


if __name__ == "__main__":
    from backend.main import configure_model_logging
    configure_model_logging()

    question = (
        "What is the effectiveness of CBT "
        "for generalized anxiety disorder?"
    )

    results = search_research(question)

    for article in results:

        print("\n-------------------------")

        print("PMID:", article["pmid"])
        print("Title:", article["title"])
        print("Year:", article["year"])
        print("Distance:", article["distance"])
        print("Abstract:", article["abstract"])
        print("URL:", article["url"])
