
import chromadb
from sentence_transformers import SentenceTransformer

from backend.database.loader import load_dataset

client = chromadb.PersistentClient(path="./cbt_chroma")

collection = client.get_or_create_collection(
    "synthetic_cbt_examples"
)

encoder = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

def index_synthetic_dataset(path: str):
    examples = load_dataset(path)

    training_examples = [
        ex for ex in examples
        if ex.metadata.split == "train"
        and ex.safety.route == "standard_cbt"
    ]

    if not training_examples:
        return 0

    documents = [ex.user_message for ex in training_examples]

    embeddings = encoder.encode(
        documents,
        normalize_embeddings=True
    ).tolist()

    collection.upsert(
        ids=[ex.case_id for ex in training_examples],
        documents=documents,
        embeddings=embeddings,
        metadatas=[
            {
                "source_type": "synthetic_cbt",
                "scenario_category": ex.scenario_category,
                "selected_intervention":
                    ex.tot.selected_intervention,
                "split": "train"
            }
            for ex in training_examples
        ]
    )

    return len(training_examples)


def retrieve_similar_cases(message: str, top_k: int = 3):
    if collection.count() == 0:
        return []

    embedding = encoder.encode(
        [message],
        normalize_embeddings=True
    ).tolist()

    results = collection.query(
        query_embeddings=embedding,
        n_results=min(top_k, collection.count())
    )

    return [
        {
            "case_id": case_id,
            "message": document,
            "metadata": metadata
        }
        for case_id, document, metadata in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0]
        )
    ]
