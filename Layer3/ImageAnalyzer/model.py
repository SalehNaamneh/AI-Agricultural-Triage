import torch
import torch.nn as nn
from torchvision import models
from pathlib import Path

from dataset import CLASSES, CLASS_NAMES_HE

MODEL_PATH = Path(__file__).parent / "best_model.pth"
NUM_CLASSES = len(CLASSES)


def build_model(pretrained: bool = True) -> nn.Module:
    weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = models.efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, NUM_CLASSES),
    )
    return model


def load_model(device: torch.device) -> nn.Module:
    model = build_model(pretrained=False)
    state = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def predict_tensor(model: nn.Module, tensor: torch.Tensor, device: torch.device) -> dict:
    tensor = tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1).squeeze().cpu().tolist()

    top_idx  = int(torch.argmax(torch.tensor(probs)))
    top_cls  = CLASSES[top_idx]
    health_idx = CLASSES.index("Healthy leaves")

    return {
        "class_en":     top_cls,
        "class_he":     CLASS_NAMES_HE.get(top_cls, top_cls),
        "confidence":   round(probs[top_idx] * 100, 1),
        "health_score": round(probs[health_idx] * 100, 1),
        "all_classes":  [
            {
                "class_en":   cls,
                "class_he":   CLASS_NAMES_HE.get(cls, cls),
                "probability": round(p * 100, 1),
            }
            for cls, p in zip(CLASSES, probs)
        ],
    }
