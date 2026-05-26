import torch
from PIL import Image
from pathlib import Path

from dataset import VAL_TRANSFORM
from model import load_model, predict_tensor

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_model = None


def get_model():
    global _model
    if _model is None:
        _model = load_model(DEVICE)
    return _model


def predict_image(image_path: str | Path) -> dict:
    img    = Image.open(image_path).convert("RGB")
    tensor = VAL_TRANSFORM(img)
    return predict_tensor(get_model(), tensor, DEVICE)


def predict_pil(image: Image.Image) -> dict:
    tensor = VAL_TRANSFORM(image)
    return predict_tensor(get_model(), tensor, DEVICE)


if __name__ == "__main__":
    import sys, io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    path = sys.argv[1] if len(sys.argv) > 1 else None
    if not path:
        print("Usage: python predict.py <image_path>")
        sys.exit(1)
    result = predict_image(path)
    print(f"Prediction  : {result['class_he']} ({result['class_en']})")
    print(f"Confidence  : {result['confidence']}%")
    print(f"Health score: {result['health_score']}%")
    print("\nAll probabilities:")
    for c in sorted(result["all_classes"], key=lambda x: x["probability"], reverse=True):
        print(f"  {c['class_en']:30s}  {c['probability']:5.1f}%")
