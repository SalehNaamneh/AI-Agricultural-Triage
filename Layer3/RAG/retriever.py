from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from ingest import CHROMA_PATH, COLLECTION_NAME, EMBEDDING_MODEL, build_index  # noqa: F401


def get_collection() -> chromadb.Collection:
    embed_fn = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    try:
        return client.get_collection(name=COLLECTION_NAME, embedding_function=embed_fn)
    except Exception:
        return build_index()


def retrieve(query: str, n_results: int = 5, doc_type: str | None = None) -> list[dict]:
    collection = get_collection()

    where = {"type": doc_type} if doc_type else None
    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    docs = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        docs.append({"content": doc, "metadata": meta, "score": 1 - dist})
    return docs


if __name__ == "__main__":
    query = "What are the symptoms of Purple Blotch and how do I treat it?"
    results = retrieve(query, n_results=4)
    for r in results:
        print(f"[{r['metadata']['type']}] score={r['score']:.3f}")
        print(r["content"][:300])
        print("---")