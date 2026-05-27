import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from ingest import CHROMA_PATH, COLLECTION_NAME, EMBEDDING_MODEL, build_index  # noqa: F401
from crop_config import load_all_crops


def _build_he_name_map() -> dict[str, str]:
    """Hebrew disease name → English disease name, from all crop configs."""
    mapping = {}
    for crop in load_all_crops().values():
        disease_info = crop.load_disease_info()
        for eng_key, row in disease_info.items():
            he = row.get("disease_name_he", "")
            if he:
                mapping[he] = eng_key
        for cls in crop.classes:
            if cls.name_he and cls.csv_key:
                mapping[cls.name_he] = cls.csv_key
    return mapping


# Built once at module load; tiny dict so this is cheap.
_HE_NAME_MAP: dict[str, str] = {}


def _disease_he_in_query(query: str) -> str | None:
    """Return the Hebrew disease name substring found in the query, or None."""
    global _HE_NAME_MAP
    if not _HE_NAME_MAP:
        _HE_NAME_MAP = _build_he_name_map()
    for he_name in _HE_NAME_MAP:
        if he_name in query:
            return he_name
    return None


_client: chromadb.ClientAPI | None = None
_collection: chromadb.Collection | None = None


def get_collection() -> chromadb.Collection:
    global _client, _collection
    if _collection is not None:
        return _collection
    embed_fn = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
    if _client is None:
        _client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    try:
        _collection = _client.get_collection(name=COLLECTION_NAME, embedding_function=embed_fn)
    except Exception:
        _collection = build_index()
    return _collection


def retrieve(query: str, n_results: int = 5, doc_type: str | None = None) -> list[dict]:
    collection = get_collection()

    # If the query mentions a known Hebrew disease name, pin the filter to it
    # so that semantic similarity can't pull in unrelated diseases.
    he_disease = _disease_he_in_query(query)
    if he_disease:
        where: dict | None = {"disease_he": {"$eq": he_disease}}
        if doc_type:
            where = {"$and": [where, {"type": doc_type}]}
    else:
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
