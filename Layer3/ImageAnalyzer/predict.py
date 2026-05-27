import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from PIL import Image

from crop_config import load_crop, CropConfig
from dataset import VAL_TRANSFORM
from model import load_model, predict_tensor

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_model_cache: dict[str, torch.nn.Module] = {}


def get_model(crop: CropConfig) -> torch.nn.Module:
    if crop.crop_id not in _model_cache:
        _model_cache[crop.crop_id] = load_model(crop, DEVICE)
    return _model_cache[crop.crop_id]


def predict_image(image_path: str | Path, crop_id: str = "onion") -> dict:
    crop   = load_crop(crop_id)
    img    = Image.open(image_path).convert("RGB")
    tensor = VAL_TRANSFORM(img)
    return predict_tensor(get_model(crop), tensor, crop, DEVICE)


def predict_pil(image: Image.Image, crop_id: str = "onion") -> dict:
    crop   = load_crop(crop_id)
    tensor = VAL_TRANSFORM(image)
    return predict_tensor(get_model(crop), tensor, crop, DEVICE)


if __name__ == "__main__":
    import sys, io, json
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Path to image file")
    parser.add_argument("--crop", default="onion")
    args = parser.parse_args()

    result = predict_image(args.image, args.crop)
    print(f"Crop        : {result['crop_he']} ({result['crop_en']})")
    print(f"Prediction  : {result['class_he']} ({result['class_en']})")
    print(f"Confidence  : {result['confidence']}%")
    print(f"Health score: {result['health_score']}%")
    print("\nAll probabilities:")
    for c in sorted(result["all_classes"], key=lambda x: x["probability"], reverse=True):
        print(f"  {c['class_en']:35s}  {c['probability']:5.1f}%")
