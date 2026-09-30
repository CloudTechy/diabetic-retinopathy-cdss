#!/usr/bin/env python3
"""
=============================================================================
APTOS 2019 — Complete Training & Evaluation Pipeline for Google Colab
PGD Computer Science, Faculty of Physical Sciences
=============================================================================

Run this entire script in a Google Colab notebook with GPU runtime.
It will:
  1. Install dependencies
  2. Download APTOS 2019 dataset from Kaggle
  3. Build a leakage-free split: hash image bytes, drop conflicting-label
     duplicate groups, keep one per group, assert zero partition overlap
  4. Train EfficientNet-B0 for 15 epochs on real retinal fundus images
  5. Evaluate on the held-out test split (leakage-free, ~525 images)
  6. Run 100-pass inference benchmark
  7. Generate learning curves and confusion matrix plots
  8. Package all artifacts for download

Prerequisites:
  - Google Colab with GPU runtime (T4 or better)
  - Kaggle API token uploaded to Colab

Usage in Colab:
  Cell 1: Upload kaggle.json
  Cell 2: !python colab_train_and_evaluate.py
"""

import subprocess
import sys
import os


def install_deps():
    """Install required packages."""
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                           "kaggle", "scikit-learn", "matplotlib", "seaborn", "pandas"])
    print("[OK] Dependencies installed.")


def setup_kaggle():
    """Configure Kaggle API credentials."""
    kaggle_dir = os.path.expanduser("~/.kaggle")
    kaggle_json = os.path.join(kaggle_dir, "kaggle.json")

    if not os.path.exists(kaggle_json):
        # Try Colab file upload
        try:
            from google.colab import files
            print("Please upload your kaggle.json file:")
            uploaded = files.upload()
            os.makedirs(kaggle_dir, exist_ok=True)
            with open(kaggle_json, "wb") as f:
                f.write(uploaded["kaggle.json"])
            os.chmod(kaggle_json, 0o600)
            print("[OK] kaggle.json uploaded and configured.")
        except ImportError:
            raise FileNotFoundError(
                "kaggle.json not found. Upload it to ~/.kaggle/kaggle.json"
            )
    else:
        os.chmod(kaggle_json, 0o600)
        print(f"[OK] kaggle.json found at {kaggle_json}")


def download_aptos():
    """Download APTOS 2019 dataset from Kaggle."""
    dataset_dir = "aptos2019"
    images_dir = os.path.join(dataset_dir, "train_images")

    if os.path.exists(images_dir) and len(os.listdir(images_dir)) > 3000:
        n = len([f for f in os.listdir(images_dir) if f.endswith(".png")])
        print(f"[OK] APTOS images already present: {n} files in {images_dir}")
        return dataset_dir

    print("Downloading APTOS 2019 from Kaggle...")
    subprocess.check_call([
        "kaggle", "competitions", "download",
        "-c", "aptos2019-blindness-detection",
        "-p", dataset_dir,
    ])

    # Unzip
    import zipfile
    zip_path = os.path.join(dataset_dir, "aptos2019-blindness-detection.zip")
    if os.path.exists(zip_path):
        print("Extracting dataset...")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(dataset_dir)
        os.remove(zip_path)

    n = len([f for f in os.listdir(images_dir) if f.endswith(".png")])
    print(f"[OK] Dataset ready: {n} images in {images_dir}")
    return dataset_dir


# ===================================================================
# STEP 1: Generate Manifest with Real Byte Hashes
# ===================================================================
def generate_manifest(dataset_dir):
    """
    Build the split via backend/scripts/build_clean_split.py - the single
    canonical implementation.

    The previous version keyed de-duplication on APTOS's duplicated_info.csv,
    which is NOT part of the Kaggle competition download. When absent it fell
    back to giving every image its own group, so the step ran and grouped
    nothing. The resulting manifest had 3,662 records over only 3,534 unique
    images, 48 duplicate groups spanning partitions, and 30 groups carrying
    conflicting labels.

    build_clean_split keys on the SHA-256 of the image bytes, which we always
    have, excludes label-conflicting groups, keeps one representative per
    group, and asserts zero hash overlap between partitions before returning.
    """
    import csv
    import json
    import sys

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    scripts_dir = os.path.join(repo_root, "backend", "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    from build_clean_split import build_clean_split, FIELDNAMES

    images_dir = os.path.join(dataset_dir, "train_images")
    labels_csv = os.path.join(dataset_dir, "train.csv")

    print("\n" + "=" * 72)
    print("STEP 1: Clean Split (de-duplicated by image bytes)")

    rows, report = build_clean_split(images_dir, labels_csv, seed=42)

    os.makedirs("output", exist_ok=True)
    manifest_path = "output/dataset_split_manifest.csv"
    with open(manifest_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    with open("output/dataset_split_audit.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    train_recs = [r for r in rows if r["split"] == "train"]
    val_recs = [r for r in rows if r["split"] == "val"]
    test_recs = [r for r in rows if r["split"] == "test"]

    print(f"Manifest saved: {manifest_path} ({len(rows)} records)")
    print(f"Audit saved:    output/dataset_split_audit.json")

    return manifest_path, images_dir, train_recs, val_recs, test_recs


# ===================================================================
# STEP 2: Train EfficientNet-B0
# ===================================================================
def train_model(manifest_path, images_dir):
    import csv
    import json
    import random
    import hashlib
    import time
    from datetime import datetime
    from collections import Counter

    import numpy as np
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    import torchvision.models as models
    import torchvision.transforms as transforms
    from PIL import Image
    from sklearn.metrics import cohen_kappa_score, f1_score

    SEED = 42
    EPOCHS = 15
    BATCH_SIZE = 32
    LR = 1e-4
    WD = 1e-4
    NUM_WORKERS = 2
    IMG_SIZE = 224

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*72}")
    print(f"STEP 2: Training EfficientNet-B0")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load manifest
    with open(manifest_path, "r") as f:
        all_records = list(csv.DictReader(f))
    train_recs = [r for r in all_records if r["split"] == "train"]
    val_recs = [r for r in all_records if r["split"] == "val"]
    print(f"Train: {len(train_recs)}, Val: {len(val_recs)}")

    # Dataset
    class APTOSDataset(Dataset):
        def __init__(self, records, img_dir, transform=None):
            self.records = records
            self.img_dir = img_dir
            self.transform = transform

        def __len__(self):
            return len(self.records)

        def __getitem__(self, idx):
            rec = self.records[idx]
            path = os.path.join(self.img_dir, f"{rec['image_id']}.png")
            img = Image.open(path).convert("RGB")
            if self.transform:
                img = self.transform(img)
            return img, int(rec["true_grade"])

    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_ds = APTOSDataset(train_recs, images_dir, train_tf)
    val_ds = APTOSDataset(val_recs, images_dir, val_tf)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=True)

    # Model
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    in_feat = model.classifier[1].in_features
    model.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(in_feat, 5))
    model.to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss, optimizer, scheduler
    counts = Counter(int(r["true_grade"]) for r in train_recs)
    n = len(train_recs)
    w = torch.tensor([n / (5 * counts[c]) for c in range(5)], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=w)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    best_qwk = -1.0
    best_epoch = 0
    ckpt_path = "output/efficientnet_b0_dr.pth"
    history = []
    log_lines = []

    def log(msg):
        print(msg)
        log_lines.append(msg)

    log(f"\n{'Ep':>3} | {'TrLoss':>8} | {'VLoss':>8} | {'VAcc':>6} | {'VQWK':>6} | {'VF1':>6} | {'LR':>10} | Status")
    log("-" * 78)

    t0 = time.time()
    for epoch in range(1, EPOCHS + 1):
        ep_start = time.time()

        # Train
        model.train()
        running_loss = 0.0
        nb = 0
        for imgs, labels in train_dl:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            nb += 1
        train_loss = running_loss / max(nb, 1)

        # Validate
        model.eval()
        vloss_sum = 0.0
        vn = 0
        preds, trues = [], []
        with torch.inference_mode():
            for imgs, labels in val_dl:
                imgs, labels = imgs.to(device), labels.to(device)
                logits = model(imgs)
                vloss_sum += criterion(logits, labels).item()
                vn += 1
                preds.extend(logits.argmax(1).cpu().tolist())
                trues.extend(labels.cpu().tolist())

        val_loss = vloss_sum / max(vn, 1)
        val_acc = sum(p == t for p, t in zip(preds, trues)) / len(trues)
        val_qwk = cohen_kappa_score(trues, preds, weights="quadratic")
        val_f1 = f1_score(trues, preds, average="macro", zero_division=0)

        lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        ep_time = time.time() - ep_start

        status = ""
        if val_qwk > best_qwk:
            best_qwk = val_qwk
            best_epoch = epoch
            torch.save(model.state_dict(), ckpt_path)
            status = f"[BEST] ({ep_time:.0f}s)"
        else:
            status = f"({ep_time:.0f}s)"

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6),
            "val_accuracy": round(val_acc, 6),
            "val_qwk": round(val_qwk, 6),
            "val_f1": round(val_f1, 6),
            "lr": f"{lr:.2e}",
        })

        log(f"{epoch:3d} | {train_loss:8.6f} | {val_loss:8.6f} | {val_acc*100:5.2f}% | "
            f"{val_qwk:6.4f} | {val_f1:6.4f} | {lr:.2e} | {status}")

    total_time = time.time() - t0
    log(f"\nDone in {total_time:.0f}s. Best: epoch {best_epoch}, QWK={best_qwk:.4f}")

    # Checkpoint hash
    with open(ckpt_path, "rb") as f:
        ckpt_hash = hashlib.sha256(f.read()).hexdigest()
    ckpt_mb = os.path.getsize(ckpt_path) / (1024 * 1024)
    log(f"Checkpoint: {ckpt_path} ({ckpt_mb:.2f} MB)")
    log(f"SHA-256: {ckpt_hash}")

    # Save history
    with open("output/epoch_history.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    # Save training log
    with open("output/training_execution.log", "w") as f:
        f.write("\n".join(log_lines))

    # Save summary
    summary = {
        "model": "EfficientNet-B0",
        "pretrained": "ImageNet (IMAGENET1K_V1)",
        "epochs": EPOCHS, "batch_size": BATCH_SIZE,
        "optimizer": "AdamW", "lr": LR, "weight_decay": WD,
        "scheduler": "CosineAnnealingLR",
        "best_epoch": best_epoch, "best_val_qwk": round(best_qwk, 6),
        "checkpoint_sha256": ckpt_hash,
        "checkpoint_size_mb": round(ckpt_mb, 2),
        "total_time_seconds": round(total_time, 1),
        "device": str(device),
        "pytorch_version": torch.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "train_samples": len(train_recs), "val_samples": len(val_recs),
    }
    with open("output/training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return ckpt_path, history


# ===================================================================
# STEP 3: Evaluate on Held-Out Test Set
# ===================================================================
def evaluate_model(manifest_path, images_dir, ckpt_path):
    import csv
    import json
    import time
    import hashlib

    import numpy as np
    import torch
    import torch.nn as nn
    import torchvision.models as models
    import torchvision.transforms as transforms
    from PIL import Image
    from sklearn.metrics import (
        cohen_kappa_score, f1_score,
        confusion_matrix as sk_cm, classification_report,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*72}")
    print(f"STEP 3: Held-Out Test Evaluation")

    # Load model
    model = models.efficientnet_b0(weights=None)
    inf = model.classifier[1].in_features
    model.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(inf, 5))
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    with open(manifest_path, "r") as f:
        test_recs = [r for r in csv.DictReader(f) if r["split"] == "test"]
    print(f"Test images: {len(test_recs)}")

    results = []
    t0 = time.time()
    with torch.inference_mode():
        for i, rec in enumerate(test_recs):
            path = os.path.join(images_dir, f"{rec['image_id']}.png")
            img = Image.open(path).convert("RGB")
            tensor = tf(img).unsqueeze(0).to(device)
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
            results.append({
                "image_id": rec["image_id"],
                "true_grade": int(rec["true_grade"]),
                "predicted_grade": int(probs.argmax()),
                **{f"score_grade_{k}": round(float(probs[k]), 6) for k in range(5)},
            })
            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{len(test_recs)}...")

    eval_time = time.time() - t0
    print(f"Inference done in {eval_time:.1f}s")

    # Save predictions
    pred_path = "output/held_out_predictions.csv"
    fnames = ["image_id", "true_grade", "predicted_grade"] + \
             [f"score_grade_{k}" for k in range(5)]
    with open(pred_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fnames)
        writer.writeheader()
        writer.writerows(results)

    # Metrics
    yt = [r["true_grade"] for r in results]
    yp = [r["predicted_grade"] for r in results]
    acc = sum(t == p for t, p in zip(yt, yp)) / len(yt)
    qwk = cohen_kappa_score(yt, yp, weights="quadratic")
    mf1 = f1_score(yt, yp, average="macro", zero_division=0)
    cm = sk_cm(yt, yp, labels=list(range(5)))

    # Argmax check
    violations = sum(
        1 for r in results
        if int(np.argmax([r[f"score_grade_{k}"] for k in range(5)])) != r["predicted_grade"]
    )

    print(f"\nAccuracy: {acc*100:.2f}% ({sum(t==p for t,p in zip(yt,yp))}/{len(yt)})")
    print(f"QWK:      {qwk:.6f}")
    print(f"Macro F1: {mf1:.6f}")
    print(f"Argmax violations: {violations}")
    print(f"\nConfusion matrix:\n{cm}")

    summary = {
        "n_test": len(results), "accuracy": round(acc, 6),
        "qwk": round(qwk, 6), "macro_f1": round(mf1, 6),
        "argmax_violations": violations,
        "confusion_matrix": cm.tolist(),
        "eval_time_s": round(eval_time, 1),
    }
    with open("output/evaluation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return cm, acc, qwk, mf1


# ===================================================================
# STEP 4: Benchmark
# ===================================================================
def run_benchmark(images_dir, ckpt_path, manifest_path):
    import csv
    import json
    import time
    import platform

    import torch
    import torch.nn as nn
    import torchvision.models as models
    import torchvision.transforms as transforms
    from PIL import Image

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*72}")
    print(f"STEP 4: 100-Run Inference Benchmark")

    model = models.efficientnet_b0(weights=None)
    inf = model.classifier[1].in_features
    model.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(inf, 5))
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    # Find a real test image
    with open(manifest_path, "r") as f:
        test_recs = [r for r in csv.DictReader(f) if r["split"] == "test"]
    img_path = os.path.join(images_dir, f"{test_recs[0]['image_id']}.png")

    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    img = Image.open(img_path).convert("RGB")
    tensor = tf(img).unsqueeze(0).to(device)

    # Warmup
    with torch.inference_mode():
        for _ in range(10):
            model(tensor)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Measure
    timings = []
    with torch.inference_mode():
        for run in range(1, 101):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                s = torch.cuda.Event(enable_timing=True)
                e = torch.cuda.Event(enable_timing=True)
                s.record()
                model(tensor)
                e.record()
                torch.cuda.synchronize()
                ms = s.elapsed_time(e)
            else:
                t0 = time.perf_counter_ns()
                model(tensor)
                ms = (time.perf_counter_ns() - t0) / 1e6
            timings.append({"run": run, "inference_time_ms": round(ms, 4)})

    vals = sorted(t["inference_time_ms"] for t in timings)
    mean = sum(vals) / len(vals)
    median = vals[len(vals) // 2]
    p95 = vals[int(0.95 * len(vals))]

    print(f"Mean: {mean:.2f} ms = {mean/1000:.5f} s")
    print(f"Median: {median:.2f} ms, P95: {p95:.2f} ms")
    print(f"Min: {min(vals):.2f} ms, Max: {max(vals):.2f} ms")

    with open("output/benchmark_timings.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["run", "inference_time_ms"])
        writer.writeheader()
        writer.writerows(timings)

    summary = {
        "scope": "forward pass only (inference_mode, batch=1)",
        "warmup": 10, "runs": 100,
        "mean_ms": round(mean, 4), "median_ms": round(median, 4),
        "p95_ms": round(p95, 4), "min_ms": round(min(vals), 4),
        "max_ms": round(max(vals), 4),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "pytorch": torch.__version__,
    }
    with open("output/benchmark_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("[OK] Benchmark complete.")


# ===================================================================
# STEP 5: Generate Plots
# ===================================================================
def generate_plots(cm, history):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    print(f"\n{'='*72}")
    print(f"STEP 5: Generating Plots")

    # Learning curves
    epochs = [h["epoch"] for h in history]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("EfficientNet-B0 Training Convergence (15 Epochs on APTOS 2019)",
                 fontsize=13, fontweight="bold")

    ax1.plot(epochs, [h["train_loss"] for h in history], "o-", label="Train Loss", color="#2563eb")
    ax1.plot(epochs, [h["val_loss"] for h in history], "s-", label="Val Loss", color="#dc2626")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training vs Validation Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, [h["val_qwk"] for h in history], "o-", label="Val QWK", color="#059669")
    ax2.plot(epochs, [h["val_f1"] for h in history], "s-", label="Val Macro F1", color="#7c3aed")
    ax2.plot(epochs, [h["val_accuracy"] for h in history], "^-", label="Val Accuracy", color="#ea580c")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Metric Value")
    ax2.set_title("Validation Metrics")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("output/learning_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[OK] Saved learning_curves.png")

    # Confusion matrix
    labels = ["No DR", "Mild\nNPDR", "Moderate\nNPDR", "Severe\nNPDR", "PDR"]
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels,
                yticklabels=labels, ax=ax, linewidths=0.5)
    ax.set_xlabel("Predicted Grade", fontsize=12)
    ax.set_ylabel("True Grade", fontsize=12)
    ax.set_title(f"Confusion Matrix — Held-Out Test Set (N={cm.sum()})", fontsize=13)
    plt.tight_layout()
    plt.savefig("output/confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[OK] Saved confusion_matrix.png")


# ===================================================================
# STEP 6: Package for Download
# ===================================================================
def package_artifacts():
    import shutil
    print(f"\n{'='*72}")
    print("STEP 6: Packaging Artifacts")

    files = [
        "output/efficientnet_b0_dr.pth",
        "output/dataset_split_manifest.csv",
        "output/held_out_predictions.csv",
        "output/epoch_history.csv",
        "output/training_execution.log",
        "output/training_summary.json",
        "output/evaluation_summary.json",
        "output/benchmark_timings.csv",
        "output/benchmark_summary.json",
        "output/learning_curves.png",
        "output/confusion_matrix.png",
    ]

    print("\nArtifacts ready for download:")
    for f in files:
        if os.path.exists(f):
            size = os.path.getsize(f)
            print(f"  [OK] {f} ({size:,} bytes)")
        else:
            print(f"  [MISSING] {f}")

    # Create zip
    shutil.make_archive("chapter4_evidence", "zip", "output")
    print(f"\n[OK] Created chapter4_evidence.zip")
    print("Download it using: from google.colab import files; files.download('chapter4_evidence.zip')")


# ===================================================================
# MAIN
# ===================================================================
def main():
    print("=" * 72)
    print("APTOS 2019 — Complete Training & Evaluation Pipeline")
    print("PGD Computer Science, Faculty of Physical Sciences")
    print("=" * 72)

    install_deps()
    setup_kaggle()
    dataset_dir = download_aptos()

    manifest_path, images_dir, _, _, _ = generate_manifest(dataset_dir)
    ckpt_path, history = train_model(manifest_path, images_dir)
    cm, acc, qwk, mf1 = evaluate_model(manifest_path, images_dir, ckpt_path)
    run_benchmark(images_dir, ckpt_path, manifest_path)
    generate_plots(cm, history)
    package_artifacts()

    print(f"\n{'='*72}")
    print("PIPELINE COMPLETE")
    print(f"{'='*72}")
    print("All empirical evidence has been generated from genuine APTOS 2019 images.")
    print("Download chapter4_evidence.zip and extract to docs/chapter4/")


if __name__ == "__main__":
    main()
