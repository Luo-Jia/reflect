
import os
import psycopg

from pgvector import Vector
import numpy as np
from pgvector.psycopg import register_vector

import logging

logger = logging.getLogger(__name__)


class VectorDatabase:

    def __init__(self):

        self.connection_string = os.getenv(
            "DATABASE_URL",
            # "postgresql://postgres:postgres@localhost:5432/cbt_db",
            "postgresql://research_user:change_this_password@localhost:5432/research_db",
        )

        self.conn = psycopg.connect(
            self.connection_string
        )

        register_vector(self.conn)

        logger.debug("Connected to PostgreSQL + pgvector")

    def create_table(self):

        with self.conn.cursor() as cursor:

            cursor.execute("""
                CREATE EXTENSION IF NOT EXISTS vector;

                CREATE TABLE IF NOT EXISTS pubmed_articles (
                    id BIGSERIAL PRIMARY KEY,
                    pmid VARCHAR(30) UNIQUE NOT NULL,
                    title TEXT,
                    authors TEXT[],
                    journal TEXT,
                    publication_year VARCHAR(20),
                    doi TEXT,
                    abstract TEXT,
                    article_url TEXT,
                    publication_types TEXT,
                    mesh_terms TEXT,
                    embedding VECTOR(384),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

        self.conn.commit()

    def insert_article(self, article: dict, embedding):

        vector = Vector(embedding.tolist())

        with self.conn.cursor() as cursor:

            cursor.execute("""
                INSERT INTO pubmed_articles (
                    pmid,
                    title,
                    authors,
                    journal,
                    publication_year,
                    doi,
                    abstract,
                    article_url,
                    publication_types,
                    mesh_terms,
                    embedding
                )
                VALUES (
                    %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s
                )

                ON CONFLICT (pmid)
                DO UPDATE SET
                    title = EXCLUDED.title,
                    authors = EXCLUDED.authors,
                    journal = EXCLUDED.journal,
                    publication_year = EXCLUDED.publication_year,
                    doi = EXCLUDED.doi,
                    abstract = EXCLUDED.abstract,
                    article_url = EXCLUDED.article_url,
                    publication_types = EXCLUDED.publication_types,
                    mesh_terms = EXCLUDED.mesh_terms,
                    embedding = EXCLUDED.embedding
            """, (
                article["pmid"],
                article.get("title"),
                article.get("authors", []),
                article.get("journal"),
                article.get("year"),
                article.get("doi"),
                article.get("abstract"),
                article.get("url"),
                article.get("publication_types", []),
                article.get("mesh_terms", []),
                vector,
            ))

        self.conn.commit()

    def search_similar(
        self,
        embedding,
        limit=5,
    ):

        vector = Vector(embedding.tolist())

        with self.conn.cursor() as cursor:

            cursor.execute("""
                SELECT
                    pmid,
                    title,
                    authors,
                    journal,
                    publication_year,
                    doi,
                    abstract,
                    article_url,
                    embedding <=> %s AS distance

                FROM pubmed_articles

                WHERE embedding IS NOT NULL

                ORDER BY embedding <=> %s

                LIMIT %s
            """, (
                vector,
                vector,
                limit,
            ))

            rows = cursor.fetchall()

        columns = [
            "pmid",
            "title",
            "authors",
            "journal",
            "year",
            "doi",
            "abstract",
            "url",
            "distance",
        ]

        return [
            dict(zip(columns, row))
            for row in rows
        ]

    def search_mmr(
        self,
        embedding,
        limit=5,
        fetch_k=30,
        lambda_mult=0.7,
    ):
    # """
    # Maximal Marginal Relevance search.

    # Args:
    #     embedding: User query embedding.
    #     limit: Number of final articles.
    #     fetch_k: Number of initial candidates.
    #     lambda_mult: Relevance/diversity balance (0-1).

    # Returns:
    #     Diverse and relevant PubMed articles.
    # """

        if not 0 <= lambda_mult <= 1:
            raise ValueError("lambda_mult must be between 0 and 1")

        if limit < 1 or fetch_k < limit:
            raise ValueError("fetch_k must be >= limit >= 1")

        query_vector = np.asarray(
            embedding,
            dtype=np.float32,
        ).reshape(-1)

        vector = Vector(query_vector.tolist())

        # Step 1: Retrieve initial candidates from pgvector
        with self.conn.cursor() as cursor:

            cursor.execute("""
                SELECT
                    pmid,
                    title,
                    authors,
                    journal,
                    publication_year,
                    doi,
                    abstract,
                    article_url,
                    publication_types,
                    mesh_terms,
                    embedding,
                    embedding <=> %s AS distance

                FROM pubmed_articles

                WHERE embedding IS NOT NULL

                ORDER BY embedding <=> %s

                LIMIT %s
            """, (
                vector,
                vector,
                fetch_k,
            ))

            rows = cursor.fetchall()

        if not rows:
            return []

        # Step 2: Convert database results into dictionaries
        columns = [
            "pmid",
            "title",
            "authors",
            "journal",
            "year",
            "doi",
            "abstract",
            "url",
            "publication_types",
            "mesh_terms",
            "embedding",
            "distance",
        ]

        candidates = [
            dict(zip(columns, row))
            for row in rows
        ]

        # Step 3: Prepare normalized vectors
        query_norm = query_vector / (
            np.linalg.norm(query_vector) + 1e-10
        )

        candidate_vectors = []

        for article in candidates:

            stored_embedding = article["embedding"]
            if isinstance(stored_embedding, Vector):
                stored_embedding = stored_embedding.to_numpy()

            article_vector = np.asarray(
                stored_embedding,
                dtype=np.float32,
            ).reshape(-1)

            norm = np.linalg.norm(article_vector)

            if norm == 0:
                article_vector = np.zeros_like(article_vector)
            else:
                article_vector = article_vector / norm

            candidate_vectors.append(article_vector)

        # Step 4: Calculate relevance scores
        relevance_scores = [
            float(np.dot(vector, query_norm))
            for vector in candidate_vectors
        ]

        # Step 5: MMR selection
        selected_indices = []
        remaining_indices = list(range(len(candidates)))

        while remaining_indices and len(selected_indices) < limit:

            best_index = None
            best_score = float("-inf")

            for index in remaining_indices:

                relevance = relevance_scores[index]

                if selected_indices:

                    redundancy = max(
                        float(
                            np.dot(
                                candidate_vectors[index],
                                candidate_vectors[selected],
                            )
                        )
                        for selected in selected_indices
                    )

                else:
                    redundancy = 0.0

                mmr_score = (
                    lambda_mult * relevance
                    - (1 - lambda_mult) * redundancy
                )

                if mmr_score > best_score:

                    best_score = mmr_score
                    best_index = index

            selected_indices.append(best_index)
            remaining_indices.remove(best_index)

        # Step 6: Return selected articles
        results = []

        for index in selected_indices:

            article = candidates[index].copy()

            article["mmr_score"] = (
                lambda_mult * relevance_scores[index]
                - (1 - lambda_mult) * max(
                    [
                        float(
                            np.dot(
                                candidate_vectors[index],
                                candidate_vectors[selected],
                            )
                        )
                        for selected in selected_indices
                        if selected != index
                        and selected_indices.index(selected)
                        < selected_indices.index(index)
                    ],
                    default=0.0,
                )
            )

            # Do not expose raw embeddings to the agent
            article.pop("embedding", None)

            results.append(article)

        return results

    def close(self):

        if self.conn:
            self.conn.close()
