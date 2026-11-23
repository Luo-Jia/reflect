
import json
import logging

from backend.ingestion.pubmed_tools import search_pubmed
from backend.model.embedding_model import EmbeddingModel
from backend.database.vector_database import VectorDatabase

logger = logging.getLogger(__spec__.name if __spec__ else __name__)

# ANSI color codes for better visual distinction
BLUE = '\033[94m'
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BOLD = '\033[1m'
RESET = '\033[0m'


def ingest_pubmed_articles(query, max_results=10):

    # Initialize components
    embedding_model = EmbeddingModel()
    database = VectorDatabase()

    database.create_table()

    try:

        # Search PubMed
        logger.info("Searching PubMed")

        results = search_pubmed(
            query=query,
            max_results=max_results,
        )

        data = json.loads(results)

        articles = data.get("articles", [])

        if not articles:
            logger.info("No PubMed articles found")
            return

        logger.info("Retrieved %d articles", len(articles))

        stored_count = 0

        # Generate embeddings and store articles
        for article in articles:

            title = article.get("title", "")
            abstract = article.get("abstract", "")

            content = f"""
            Title: {title}

            Journal: {article.get('journal', '')}

            Year: {article.get('year', '')}

            Abstract: {abstract}
            """

            if not title.strip() and not abstract.strip():
                continue

            embedding = embedding_model.generate_embedding(
                content
            )

            database.insert_article(
                article,
                embedding,
            )

            stored_count += 1

            logger.debug("Stored publication PMID: %s", article["pmid"])

        logger.info("PubMed ingestion completed: retrieved=%d stored=%d", len(articles), stored_count)

    finally:
        database.close()


if __name__ == "__main__":
    from backend.main import configure_model_logging
    configure_model_logging()

    query = (
        '"Cognitive Behavioral Therapy"[Title/Abstract] '
        'AND "Anxiety"[Title/Abstract]'
    )

    ingest_pubmed_articles(
        query=query,
        max_results=10,
    )
