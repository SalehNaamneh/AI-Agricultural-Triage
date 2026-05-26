from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, WeightedRandomSampler
from torchvision import transforms
import numpy as np

IMAGES_DIR = Path(__file__).resolve().parents[2] / "data" / "data" / "onion" / "images"

# Sorted so class index is deterministic across runs
CLASSES = sorted([d.name for d in IMAGES_DIR.iterdir() if d.is_dir()])

CLASS_TO_IDX = {cls: i for i, cls in enumerate(CLASSES)}

CLASS_NAMES_HE = {
    "Bulb Rot":                "ריקבון הבצל",
    "Bulb_blight-D":           "כימשון הבצלת",
    "Caterpillar-P":           "זחל (מזיק)",
    "Downy mildew":            "כימשון הבצל",
    "Fusarium-D":              "פוזריום",
    "Healthy leaves":          "בצל בריא",
    "Purple blotch":           "כתם סגול",
    "Rust":                    "חלודה",
    "stemphylium Leaf Blight": "סטמפיליום",
    "Virosis-D":               "וירוס",
}

TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

VAL_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class OnionDataset(Dataset):
    def __init__(self, samples: list[tuple[Path, int]], transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def load_samples() -> list[tuple[Path, int]]:
    samples = []
    for cls in CLASSES:
        folder = IMAGES_DIR / cls
        for img_path in folder.glob("*.jpg"):
            samples.append((img_path, CLASS_TO_IDX[cls]))
    return samples


def make_weighted_sampler(samples: list[tuple[Path, int]]) -> WeightedRandomSampler:
    labels = np.array([s[1] for s in samples])
    class_counts = np.bincount(labels, minlength=len(CLASSES))
    class_weights = 1.0 / np.maximum(class_counts, 1)
    sample_weights = class_weights[labels]
    return WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).float(),
        num_samples=len(samples),
        replacement=True,
    )


def train_val_split(samples, val_ratio=0.2, seed=42):
    rng = np.random.default_rng(seed)
    # Stratified split per class
    train, val = [], []
    labels = np.array([s[1] for s in samples])
    for cls_idx in range(len(CLASSES)):
        idx = np.where(labels == cls_idx)[0]
        rng.shuffle(idx)
        split = max(1, int(len(idx) * val_ratio))
        val   += [samples[i] for i in idx[:split]]
        train += [samples[i] for i in idx[split:]]
    return train, val
