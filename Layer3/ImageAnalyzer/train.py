import sys
import argparse
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from crop_config import load_crop, load_all_crops
from dataset import CropDataset, load_samples, make_weighted_sampler, train_val_split, TRAIN_TRANSFORM, VAL_TRANSFORM
from model import build_model

EPOCHS    = 30
LR        = 1e-3
PATIENCE  = 5
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(crop_id: str, batch_size: int = 16):
    crop = load_crop(crop_id)
    print(f"Crop     : {crop.name_he} ({crop.name_en})")
    print(f"Device   : {DEVICE}")
    print(f"Classes  : {crop.num_classes} → {crop.class_folders}\n")

    all_samples = load_samples(crop)
    if not all_samples:
        raise RuntimeError(f"No images found for crop '{crop_id}'. Check {crop.images_dir}")

    train_samples, val_samples = train_val_split(all_samples)
    print(f"Train: {len(train_samples)} | Val: {len(val_samples)}")

    train_ds = CropDataset(train_samples, transform=TRAIN_TRANSFORM)
    val_ds   = CropDataset(val_samples,   transform=VAL_TRANSFORM)

    sampler      = make_weighted_sampler(train_samples, crop.num_classes)
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=4, pin_memory=True)

    model = build_model(num_classes=crop.num_classes, pretrained=True).to(DEVICE)

    labels_arr = [s[1] for s in train_samples]
    counts = np.bincount(labels_arr, minlength=crop.num_classes)
    cw = torch.tensor(1.0 / np.maximum(counts, 1), dtype=torch.float32).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=cw)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_acc, no_improve = 0.0, 0

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

        model.train()
        train_loss, train_correct = 0.0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            out  = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            train_loss    += loss.item() * imgs.size(0)
            train_correct += (out.argmax(1) == labels).sum().item()

        model.eval()
        val_loss, val_correct = 0.0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                out = model(imgs)
                val_loss    += criterion(out, labels).item() * imgs.size(0)
                val_correct += (out.argmax(1) == labels).sum().item()

        scheduler.step()

        train_acc = train_correct / len(train_samples) * 100
        val_acc   = val_correct   / len(val_samples)   * 100

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Train loss {train_loss/len(train_samples):.4f} acc {train_acc:.1f}% | "
            f"Val loss {val_loss/len(val_samples):.4f} acc {val_acc:.1f}% | "
            f"{time.time()-t0:.0f}s"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), crop.model_path)
            print(f"  >> Saved best model (val_acc={val_acc:.1f}%) -> {crop.model_path}")
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"Early stopping after {epoch} epochs.")
                break

    print(f"\nDone. Best val accuracy: {best_val_acc:.1f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--crop", default="onion",
                        help="Crop ID matching a folder in data/data/ (default: onion)")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    train(args.crop, args.batch_size)
