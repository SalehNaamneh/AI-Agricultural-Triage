import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.nn as nn
from torchvision import models

from crop_config import CropConfig


def build_model(num_classes: int, pretrained: bool = True) -> nn.Module:
    weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = models.efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes),
    )
    return model


def load_model(crop: CropConfig, device: torch.device) -> nn.Module:
    if not crop.model_path.exists():
        raise FileNotFoundError(
            f"No trained model found for crop '{crop.crop_id}' at {crop.model_path}\n"
            f"Run: python train.py --crop {crop.crop_id}"
        )
    model = build_model(num_classes=crop.num_classes, pretrained=False)
    state = torch.load(crop.model_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def predict_tensor(model: nn.Module, tensor: torch.Tensor,
                   crop: CropConfig, device: torch.device) -> dict:
    tensor = tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1).squeeze().cpu().tolist()

    top_idx    = int(torch.argmax(torch.tensor(probs)))
    top_folder = crop.idx_to_folder[top_idx]
    top_class  = crop.folder_to_class[top_folder]

    healthy_folder = next(
        (c.folder for c in crop.classes if c.csv_key and "healthy" in c.csv_key.lower()),
        None,
    )
    health_score = 0.0
    if healthy_folder and healthy_folder in crop.folder_to_idx:
        health_score = round(probs[crop.folder_to_idx[healthy_folder]] * 100, 1)

    return {
        "crop_en":      crop.name_en,
        "crop_he":      crop.name_he,
        "class_en":     top_folder,
        "class_he":     top_class.name_he,
        "confidence":   round(probs[top_idx] * 100, 1),
        "health_score": health_score,
        "all_classes": [
            {
                "class_en":    crop.idx_to_folder[i],
                "class_he":    crop.folder_to_class[crop.idx_to_folder[i]].name_he,
                "probability": round(p * 100, 1),
            }
            for i, p in enumerate(probs)
        ],
    }
