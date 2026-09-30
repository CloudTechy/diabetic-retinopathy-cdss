#!/usr/bin/env python3
"""
EfficientNet-B0 Training Script — Genuine Image-Based Training
PGD Computer Science, Faculty of Physical Sciences

Trains on ACTUAL APTOS 2019 retinal fundus images using:
- torchvision EfficientNet-B0 with ImageNet pretrained weights (transfer learning)
- Proper Dataset/DataLoader with real image I/O
- Class-weighted CrossEntropyLoss
- AdamW + CosineAnnealingLR
- Real validation metrics (loss, accuracy, QWK, F1) from genuine predictions
"""

import os
import csv
import sys
import time
import json
import random
import hashlib
import logging
from pathlib import Path
from collections import Counter
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from sklearn.metrics import (
    cohen_kappa_score,
    f1_score,
    confusion_matrix as sk_confusion_matrix,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SEED = 42
EPOCHS = 15
BATCH_SIZE = 32
LR = 1e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 4
IMAGE_SIZE = 224
NUM_CLASSES = 5


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class APTOSDataset(Dataset):
    """Reads actual APTOS 2019 retinal fundus images from disk."""

    def __init__(self, records: list, images_dir: str, transform=None):
        self.records = records
        self.images_dir = Path(images_dir)
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        img_id = rec["image_id"]
        label = int(rec["true_grade"])

        img_path = self.images_dir / f"{img_id}.png"
        if not img_path.exists():
            # Try .jpg fallback
            img_path = self.images_dir / f"{img_id}.jpg"

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, label, img_id


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------
def get_train_transforms():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def get_val_transforms():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_qwk(y_true, y_pred, num_classes=5):
    """Quadratic Weighted Kappa using scikit-learn."""
    return cohen_kappa_score(y_true, y_pred, weights="quadratic")


def compute_macro_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average="macro", zero_division=0)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_model():
    seed_everything(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 72)
    print("PGD Computer Science — EfficientNet-B0 Training on APTOS 2019")
    print(f"Device: {device}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 72)

    # --- Paths ---
    root = Path(__file__).resolve().parents[2]
    manifest_path = root / "docs" / "chapter4" / "dataset_split_manifest.csv"
    images_dir = root / "storage" / "datasets" / "aptos2019" / "train_images"
    weights_dir = root / "backend" / "models" / "weights"
    output_dir = root / "docs" / "chapter4"
    weights_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"{manifest_path} not found. Run generate_aptos_manifest.py first."
        )

    # --- Load manifest ---
    with open(manifest_path, "r", encoding="utf-8") as f:
        all_records = list(csv.DictReader(f))

    train_recs = [r for r in all_records if r["split"] == "train"]
    val_recs = [r for r in all_records if r["split"] == "val"]

    print(f"\nManifest loaded: Train={len(train_recs)}, Val={len(val_recs)}")

    # Verify at least some images exist
    sample_path = images_dir / f"{train_recs[0]['image_id']}.png"
    if not sample_path.exists():
        raise FileNotFoundError(
            f"Image {sample_path} not found. Ensure APTOS 2019 images are in {images_dir}"
        )

    # --- Datasets & DataLoaders ---
    train_dataset = APTOSDataset(train_recs, images_dir, get_train_transforms())
    val_dataset = APTOSDataset(val_recs, images_dir, get_val_transforms())

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # --- Model ---
    print("\nConstructing EfficientNet-B0 with ImageNet pretrained weights...")
    model = models.efficientnet_b0(
        weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
    )
    in_features = model.classifier[1].in_features  # 1280
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=False),
        nn.Linear(in_features, NUM_CLASSES),
    )
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params:,}  Trainable: {trainable_params:,}")

    # --- Class-weighted loss ---
    train_counts = Counter(int(r["true_grade"]) for r in train_recs)
    n_train = len(train_recs)
    class_weights = [n_train / (NUM_CLASSES * train_counts[c]) for c in range(NUM_CLASSES)]
    weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)

    print("\nClass weights (inverse frequency):")
    for c in range(NUM_CLASSES):
        print(f"  Grade {c} (N={train_counts[c]}): weight={class_weights[c]:.4f}")

    # --- Optimizer & Scheduler ---
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    # --- Training loop ---
    best_val_qwk = -1.0
    best_epoch = 0
    checkpoint_path = weights_dir / "efficientnet_b0_dr.pth"
    history = []

    log_path = output_dir / "training_execution.log"
    log_file = open(log_path, "w", encoding="utf-8")

    def log(msg):
        print(msg)
        log_file.write(msg + "\n")
        log_file.flush()

    log(f"\nTraining started at {datetime.now().isoformat()}")
    log(f"{'Epoch':>5} | {'Train Loss':>10} | {'Val Loss':>8} | {'Val Acc':>7} | "
        f"{'Val QWK':>7} | {'Val F1':>7} | {'LR':>10} | Status")
    log("-" * 80)

    training_start = time.time()

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()

        # --- Train phase ---
        model.train()
        running_loss = 0.0
        n_batches = 0

        for images, labels, _ in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            n_batches += 1

        avg_train_loss = running_loss / max(1, n_batches)

        # --- Validation phase ---
        model.eval()
        val_loss_accum = 0.0
        val_batches = 0
        all_preds = []
        all_labels = []

        with torch.inference_mode():
            for images, labels, _ in val_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                logits = model(images)
                loss = criterion(logits, labels)
                val_loss_accum += loss.item()
                val_batches += 1

                preds = logits.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds.tolist())
                all_labels.extend(labels.cpu().numpy().tolist())

        avg_val_loss = val_loss_accum / max(1, val_batches)
        val_acc = sum(p == t for p, t in zip(all_preds, all_labels)) / len(all_labels)
        val_qwk = compute_qwk(all_labels, all_preds)
        val_f1 = compute_macro_f1(all_labels, all_preds)

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        epoch_time = time.time() - epoch_start

        # --- Checkpoint ---
        status = ""
        if val_qwk > best_val_qwk:
            best_val_qwk = val_qwk
            best_epoch = epoch
            torch.save(model.state_dict(), checkpoint_path)
            status = f"[BEST] saved ({epoch_time:.1f}s)"
        else:
            status = f"({epoch_time:.1f}s)"

        history.append({
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 6),
            "val_loss": round(avg_val_loss, 6),
            "val_accuracy": round(val_acc, 6),
            "val_qwk": round(val_qwk, 6),
            "val_f1": round(val_f1, 6),
            "lr": f"{current_lr:.2e}",
        })

        log(f"{epoch:5d} | {avg_train_loss:10.6f} | {avg_val_loss:8.6f} | "
            f"{val_acc*100:6.2f}% | {val_qwk:7.4f} | {val_f1:7.4f} | "
            f"{current_lr:.2e} | {status}")

    total_time = time.time() - training_start
    log(f"\nTraining completed in {total_time:.1f}s ({total_time/60:.1f} min)")
    log(f"Best epoch: {best_epoch} (Val QWK = {best_val_qwk:.4f})")

    # --- Checkpoint verification ---
    with open(checkpoint_path, "rb") as f:
        ckpt_hash = hashlib.sha256(f.read()).hexdigest()
    ckpt_size_mb = checkpoint_path.stat().st_size / (1024 * 1024)

    log(f"\nCheckpoint: {checkpoint_path}")
    log(f"Size: {ckpt_size_mb:.2f} MB")
    log(f"SHA-256: {ckpt_hash}")
    log_file.close()

    # --- Write epoch_history.csv ---
    history_path = output_dir / "epoch_history.csv"
    with open(history_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
    print(f"\nSaved epoch history to {history_path}")

    # --- Write training summary ---
    summary = {
        "model": "EfficientNet-B0",
        "pretrained": "ImageNet (EfficientNet_B0_Weights.IMAGENET1K_V1)",
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "optimizer": "AdamW",
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "scheduler": "CosineAnnealingLR",
        "best_epoch": best_epoch,
        "best_val_qwk": best_val_qwk,
        "checkpoint_sha256": ckpt_hash,
        "checkpoint_size_mb": round(ckpt_size_mb, 2),
        "total_training_time_seconds": round(total_time, 1),
        "device": str(device),
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        "train_samples": len(train_recs),
        "val_samples": len(val_recs),
        "seed": SEED,
    }
    summary_path = output_dir / "training_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved training summary to {summary_path}")

    return history, best_epoch, best_val_qwk, ckpt_hash


if __name__ == "__main__":
    train_model()
