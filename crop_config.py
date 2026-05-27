"""
Shared crop configuration loader.
All services (RAG, ImageAnalyzer, UI) import from here.

To add a new crop:
  1. Create data/data/<crop_id>/crop.yaml
  2. Add images under data/data/<crop_id>/images/<disease_folder>/
  3. Add CSVs under data/data/<crop_id>/disease_info/
  4. Run Layer3/RAG/ingest.py  (rebuilds knowledge base for all crops)
  5. Run Layer3/ImageAnalyzer/train.py --crop <crop_id>  (trains new model)

To add a new disease to an existing crop:
  1. Add folder + images under that crop's images/ directory
  2. Add the entry in crop.yaml under classes:
  3. Optionally add rows to disease_csv and spray_csv
  4. Re-run ingest.py and train.py --crop <crop_id>
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import csv
import yaml

DATA_ROOT = Path(__file__).resolve().parent / "data" / "data"


@dataclass
class DiseaseClass:
    folder: str
    name_he: str
    csv_key: Optional[str]

    @property
    def name_en(self) -> str:
        return self.folder


@dataclass
class CropConfig:
    crop_id: str
    name_en: str
    name_he: str
    icon: str
    crop_dir: Path
    _disease_csv_rel: str
    _spray_csv_rel: str
    classes: list[DiseaseClass]

    # ── Paths ──────────────────────────────────────────────────────────────

    @property
    def images_dir(self) -> Path:
        return self.crop_dir / "images"

    @property
    def disease_csv(self) -> Path:
        return self.crop_dir / self._disease_csv_rel

    @property
    def spray_csv(self) -> Path:
        return self.crop_dir / self._spray_csv_rel

    # ── Derived helpers ────────────────────────────────────────────────────

    @property
    def class_folders(self) -> list[str]:
        """Sorted list of folder names — determines model class indices."""
        return sorted(c.folder for c in self.classes)

    @property
    def folder_to_class(self) -> dict[str, DiseaseClass]:
        return {c.folder: c for c in self.classes}

    @property
    def folder_to_idx(self) -> dict[str, int]:
        return {f: i for i, f in enumerate(self.class_folders)}

    @property
    def idx_to_folder(self) -> dict[int, str]:
        return {i: f for i, f in enumerate(self.class_folders)}

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    # ── CSV loaders ────────────────────────────────────────────────────────

    def _read_csv(self, path: Path) -> list[dict]:
        with open(path, encoding="utf-8-sig") as f:
            return list(csv.DictReader(f))

    def load_disease_info(self) -> dict[str, dict]:
        """Returns {disease_name_en: row_dict}."""
        if not self.disease_csv.exists():
            return {}
        return {r["disease_name_en"]: r for r in self._read_csv(self.disease_csv)}

    def load_treatments(self) -> dict[str, list[dict]]:
        """Returns {disease_name_en: [row_dict, ...]}."""
        if not self.spray_csv.exists():
            return {}
        result: dict[str, list] = {}
        for r in self._read_csv(self.spray_csv):
            result.setdefault(r["disease_name_en"], []).append(r)
        return result

    # ── Model path ─────────────────────────────────────────────────────────

    @property
    def model_path(self) -> Path:
        models_dir = Path(__file__).resolve().parent / "Layer3" / "ImageAnalyzer" / "models"
        models_dir.mkdir(exist_ok=True)
        return models_dir / f"{self.crop_id}_best_model.pth"


# ── Loaders ────────────────────────────────────────────────────────────────────

def load_crop(crop_id: str) -> CropConfig:
    crop_dir = DATA_ROOT / crop_id
    yaml_path = crop_dir / "crop.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"No crop.yaml found for crop '{crop_id}' at {yaml_path}")

    with open(yaml_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    classes = [
        DiseaseClass(
            folder=c["folder"],
            name_he=c["name_he"],
            csv_key=c.get("csv_key"),
        )
        for c in data["classes"]
    ]

    return CropConfig(
        crop_id=crop_id,
        name_en=data["name_en"],
        name_he=data["name_he"],
        icon=data.get("icon", "🌱"),
        crop_dir=crop_dir,
        _disease_csv_rel=data.get("disease_csv", "disease_info/diseases.csv"),
        _spray_csv_rel=data.get("spray_csv", "disease_info/spray_products.csv"),
        classes=classes,
    )


def load_all_crops() -> dict[str, CropConfig]:
    """Scans DATA_ROOT for all directories containing crop.yaml."""
    crops = {}
    if not DATA_ROOT.exists():
        return crops
    for crop_dir in sorted(DATA_ROOT.iterdir()):
        if crop_dir.is_dir() and (crop_dir / "crop.yaml").exists():
            crops[crop_dir.name] = load_crop(crop_dir.name)
    return crops
