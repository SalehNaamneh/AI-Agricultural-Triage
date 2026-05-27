import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from PIL import Image
import torch
from torch.utils.data import Dataset, WeightedRandomSampler
from torchvision import transforms
import numpy as np

from crop_config import CropConfig

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


class CropDataset(Dataset):
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


def load_samples(crop: CropConfig) -> list[tuple[Path, int]]:
    samples = []
    for folder in crop.class_folders:
        folder_path = crop.images_dir / folder
        if not folder_path.exists():
            continue
        label = crop.folder_to_idx[folder]
        for img_path in folder_path.glob("*.jpg"):
            samples.append((img_path, label))
    return samples


def make_weighted_sampler(samples: list[tuple[Path, int]], num_classes: int) -> WeightedRandomSampler:
    labels = np.array([s[1] for s in samples])
    counts = np.bincount(labels, minlength=num_classes)
    weights = 1.0 / np.maximum(counts, 1)
    sample_weights = weights[labels]
    return WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).float(),
        num_samples=len(samples),
        replacement=True,
    )


def train_val_split(samples: list, val_ratio: float = 0.2, seed: int = 42):
    rng = np.random.default_rng(seed)
    labels = np.array([s[1] for s in samples])
    num_classes = labels.max() + 1
    train, val = [], []
    for cls_idx in range(num_classes):
        idx = np.where(labels == cls_idx)[0]
        rng.shuffle(idx)
        split = max(1, int(len(idx) * val_ratio))
        val   += [samples[i] for i in idx[:split]]
        train += [samples[i] for i in idx[split:]]
    return train, val
