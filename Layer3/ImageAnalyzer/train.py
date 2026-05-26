import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import time

from dataset import (
    OnionDataset, load_samples, make_weighted_sampler,
    train_val_split, TRAIN_TRANSFORM, VAL_TRANSFORM, CLASSES,
)
from model import build_model, MODEL_PATH

# ─── Config ───────────────────────────────────────────────────────────────────
BATCH_SIZE  = 16
EPOCHS      = 30
LR          = 1e-3
PATIENCE    = 5          # early stopping patience
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train():
    print(f"Device: {DEVICE}")
    print(f"Classes ({len(CLASSES)}): {CLASSES}\n")

    all_samples = load_samples()
    train_samples, val_samples = train_val_split(all_samples)
    print(f"Train: {len(train_samples)} | Val: {len(val_samples)}")

    train_ds = OnionDataset(train_samples, transform=TRAIN_TRANSFORM)
    val_ds   = OnionDataset(val_samples,   transform=VAL_TRANSFORM)

    sampler = make_weighted_sampler(train_samples)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=4, pin_memory=True)

    model = build_model(pretrained=True).to(DEVICE)

    # Class-weighted loss to further combat imbalance
    import numpy as np
    labels = [s[1] for s in train_samples]
    counts = np.bincount(labels, minlength=len(CLASSES))
    cw = torch.tensor(1.0 / np.maximum(counts, 1), dtype=torch.float32).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=cw)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_acc = 0.0
    no_improve   = 0

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

        # ── Train ──────────────────────────────────────────────────────────
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

        # ── Validate ───────────────────────────────────────────────────────
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
        elapsed   = time.time() - t0

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Train loss {train_loss/len(train_samples):.4f} acc {train_acc:.1f}% | "
            f"Val loss {val_loss/len(val_samples):.4f} acc {val_acc:.1f}% | "
            f"{elapsed:.0f}s"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"  >> Saved best model (val_acc={val_acc:.1f}%)")
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"Early stopping after {epoch} epochs.")
                break

    print(f"\nTraining complete. Best val accuracy: {best_val_acc:.1f}%")
    print(f"Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    train()
