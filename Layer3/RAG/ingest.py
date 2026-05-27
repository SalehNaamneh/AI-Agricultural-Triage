import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from crop_config import load_all_crops, CropConfig

CHROMA_PATH     = Path(__file__).parent / "chroma_db"
COLLECTION_NAME = "agritriage_knowledge"
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"


def _disease_text_en(row: dict, crop: CropConfig) -> str:
    return (
        f"Crop: {crop.name_en}\n"
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


def _disease_text_he(row: dict, crop: CropConfig) -> str:
    return (
        f"גידול: {crop.name_he}\n"
        f"מחלה: {row['disease_name_he']}\n"
        f"תיאור: {row['description_he']}\n"
        f"תסמינים חזותיים: {row['visual_symptoms_he']}\n"
        f"סימן אבחנתי: {row['diagnostic_sign_he']}\n"
        f"תנאים מעדיפים: {row['favorable_conditions_he']}\n"
        f"מניעה: {row['agronomic_prevention_he']}\n"
        f"סיכון בלבול: {row['confusion_risk_he']}\n"
        # also include English name so the LLM can cross-reference
        f"Disease (English): {row['disease_name_en']}"
    )


def _product_text_en(row: dict, crop: CropConfig) -> str:
    return (
        f"Crop: {crop.name_en}\n"
        f"Treatment for: {row['disease_name_en']}\n"
        f"Product: {row['product_name_en']}\n"
        f"Active ingredient: {row['active_ingredient_en']}\n"
        f"FRAC code: {row['frac_code']}\n"
        f"Chemical group: {row['chemical_group_en']}\n"
        f"Product type: {row['product_type_en']}\n"
        f"Dose per dunam: {row['dose_per_dunam']}\n"
        f"Spray season: {row['spray_season_israel']}\n"
        f"Mode of action: {row['mode_of_action_en']}\n"
        f"Resistance warning: {row['resistance_warning']}"
    )


def _product_text_he(row: dict, crop: CropConfig) -> str:
    return (
        f"גידול: {crop.name_he}\n"
        f"טיפול עבור: {row['disease_name_he']}\n"
        f"תכשיר: {row['product_name_he']}\n"
        f"חומר פעיל: {row['active_ingredient_he']}\n"
        f"קוד FRAC: {row['frac_code']}\n"
        f"קבוצה כימית: {row['chemical_group_he']}\n"
        f"אופן פעולה: {row['mode_of_action_he']}\n"
        f"מינון לדונם: {row['dose_per_dunam']}\n"
        # also include English product name so the LLM can cross-reference
        f"Product (English): {row['product_name_en']}\n"
        f"Treatment for (English): {row['disease_name_en']}"
    )


def build_index(reset: bool = False) -> chromadb.Collection:
    embed_fn = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
    client   = chromadb.PersistentClient(path=str(CHROMA_PATH))

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

    crops = load_all_crops()
    if not crops:
        print("No crops found. Make sure data/data/<crop>/crop.yaml exists.")
        return collection

    documents, metadatas, ids = [], [], []
    total_diseases = total_products = 0

    for crop_id, crop in crops.items():
        disease_rows = list(crop.load_disease_info().values())
        product_rows = [r for rows in crop.load_treatments().values() for r in rows]

        print(f"Indexing {crop.name_en}: {len(disease_rows)} diseases, {len(product_rows)} products")

        base_meta = {"crop": crop_id, "crop_he": crop.name_he}

        for row in disease_rows:
            class_id = row["class_id"]
            shared = {**base_meta, "type": "disease",
                      "disease_en": row["disease_name_en"],
                      "disease_he": row["disease_name_he"]}

            # English document
            documents.append(_disease_text_en(row, crop))
            metadatas.append({**shared, "lang": "en"})
            ids.append(f"{crop_id}_disease_{class_id}_en")

            # Hebrew document
            documents.append(_disease_text_he(row, crop))
            metadatas.append({**shared, "lang": "he"})
            ids.append(f"{crop_id}_disease_{class_id}_he")

        for i, row in enumerate(product_rows):
            class_id = row["class_id"]
            shared = {**base_meta, "type": "treatment",
                      "disease_en": row["disease_name_en"],
                      "disease_he": row["disease_name_he"],
                      "product_en": row["product_name_en"],
                      "frac_code":  row["frac_code"]}

            # English document
            documents.append(_product_text_en(row, crop))
            metadatas.append({**shared, "lang": "en"})
            ids.append(f"{crop_id}_product_{class_id}_{i}_en")

            # Hebrew document
            documents.append(_product_text_he(row, crop))
            metadatas.append({**shared, "lang": "he"})
            ids.append(f"{crop_id}_product_{class_id}_{i}_he")

        total_diseases += len(disease_rows)
        total_products += len(product_rows)

    collection.add(documents=documents, metadatas=metadatas, ids=ids)
    print(f"\nIndexed {len(ids)} total documents across {len(crops)} crop(s).")
    print(f"  Diseases: {total_diseases} × 2 langs  |  Products: {total_products} × 2 langs")
    return collection


if __name__ == "__main__":
    build_index(reset=True)
