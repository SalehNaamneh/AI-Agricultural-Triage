import csv
import os
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "data" / "onion" / "disease_info"
DISEASES_CSV = DATA_DIR / "onion_diseases.csv"
PRODUCTS_CSV = DATA_DIR / "onion_spray_products.csv"

CHROMA_PATH = Path(__file__).parent / "chroma_db"
COLLECTION_NAME = "onion_agriculture"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def _read_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _disease_to_text(row: dict) -> str:
    return (
        f"Disease: {row['disease_name_en']} ({row['scientific_name']})\n"
        f"Pathogen type: {row['pathogen_type_en']}\n"
        f"Description: {row['description_en']}\n"
        f"Visual symptoms: {row['visual_symptoms_en']}\n"
        f"Diagnostic sign: {row['diagnostic_sign_en']}\n"
        f"Favorable conditions: {row['favorable_conditions_en']}\n"
        f"Prevention: {row['agronomic_prevention_en']}\n"
        f"Spray season (Israel): {row['spray_season_israel']}\n"
        f"Confusion risk: {row['confusion_risk_en']}"
    )


def _product_to_text(row: dict) -> str:
    return (
        f"Treatment for: {row['disease_name_en']}\n"
        f"Product: {row['product_name_en']} (Hebrew: {row['product_name_he']})\n"
        f"Active ingredient: {row['active_ingredient_en']}\n"
        f"FRAC code: {row['frac_code']}\n"
        f"Chemical group: {row['chemical_group_en']}\n"
        f"Product type: {row['product_type_en']}\n"
        f"Dose per dunam: {row['dose_per_dunam']}\n"
        f"Spray season: {row['spray_season_israel']}\n"
        f"Mode of action: {row['mode_of_action_en']}\n"
        f"Resistance warning: {row['resistance_warning']}"
    )


def build_index(reset: bool = False) -> chromadb.Collection:
    embed_fn = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))

    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )

    if collection.count() > 0 and not reset:
        print(f"Collection already has {collection.count()} documents. Use reset=True to rebuild.")
        return collection

    documents, metadatas, ids = [], [], []

    for row in _read_csv(DISEASES_CSV):
        doc_id = f"disease_{row['class_id']}"
        documents.append(_disease_to_text(row))
        metadatas.append({
            "type": "disease",
            "disease_en": row["disease_name_en"],
            "disease_he": row["disease_name_he"],
            "class_id": row["class_id"],
        })
        ids.append(doc_id)

    for i, row in enumerate(_read_csv(PRODUCTS_CSV)):
        doc_id = f"product_{row['class_id']}_{i}"
        documents.append(_product_to_text(row))
        metadatas.append({
            "type": "treatment",
            "disease_en": row["disease_name_en"],
            "product_en": row["product_name_en"],
            "frac_code": row["frac_code"],
            "class_id": row["class_id"],
        })
        ids.append(doc_id)

    collection.add(documents=documents, metadatas=metadatas, ids=ids)
    print(f"Indexed {len(ids)} documents ({len(_read_csv(DISEASES_CSV))} diseases, {len(_read_csv(PRODUCTS_CSV))} products).")
    return collection


if __name__ == "__main__":
    build_index(reset=True)