
from sentence_transformers import SentenceTransformer

import logging

logger = logging.getLogger(__name__)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class EmbeddingModel:

    def __init__(self):

        self.model = SentenceTransformer(
            MODEL_NAME,
            device="cpu",
        )

        logger.info("Embedding model loaded: %s", MODEL_NAME)

    def generate_embedding(self, text: str):

        if not text or not text.strip():
            raise ValueError("Text cannot be empty.")

        embedding = self.model.encode(
            text,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        return embedding

    def generate_embeddings(self, texts: list[str]):

        return self.model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            batch_size=32,
        )
