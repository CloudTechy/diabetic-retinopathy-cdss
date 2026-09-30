#!/usr/bin/env python3
"""
EfficientNet-B0 Held-Out Test Evaluation — Genuine Model Inference
PGD Computer Science, Faculty of Physical Sciences

Loads the trained checkpoint, runs inference on every held-out test image,
records per-image logits/scores/predictions, and computes all metrics
exclusively from the resulting CSV.
"""

import os
import csv
import sys
import json
import hashlib
import time
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from sklearn.metrics import (
    cohen_kappa_score,
    f1_score,
    confusion_matrix as sk_confusion_matrix,
    classification_report,
)

NUM_CLASSES = 5
IMAGE_SIZE = 224


def get_test_transforms():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def load_model(checkpoint_path: str, device: torch.device) -> nn.Module:
    """Load EfficientNet-B0 with trained weights."""
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=False),
        nn.Linear(in_features, NUM_CLASSES),
    )
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 72)
    print("PGD Computer Science — EfficientNet-B0 Held-Out Test Evaluation")
    print(f"Device: {device}")
    print(f"PyTorch: {torch.__version__}")
    print("=" * 72)

    # --- Paths ---
    root = Path(__file__).resolve().parents[2]
    manifest_path = root / "docs" / "chapter4" / "dataset_split_manifest.csv"
    images_dir = root / "storage" / "datasets" / "aptos2019" / "train_images"
    checkpoint_path = root / "backend" / "models" / "weights" / "efficientnet_b0_dr.pth"
    output_dir = root / "docs" / "chapter4"
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Verify checkpoint ---
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    with open(checkpoint_path, "rb") as f:
        ckpt_hash = hashlib.sha256(f.read()).hexdigest()
    ckpt_size_mb = checkpoint_path.stat().st_size / (1024 * 1024)
    print(f"\nCheckpoint: {checkpoint_path}")
    print(f"Size: {ckpt_size_mb:.2f} MB")
    print(f"SHA-256: {ckpt_hash}")

    # --- Load model ---
    print("\nLoading model...")
    model = load_model(str(checkpoint_path), device)
    print("Model loaded successfully. mode=eval")

    # --- Load manifest ---
    with open(manifest_path, "r", encoding="utf-8") as f:
        all_records = list(csv.DictReader(f))
    test_recs = [r for r in all_records if r["split"] == "test"]
    print(f"\nHeld-out test samples: {len(test_recs)}")

    # --- Run inference on every test image ---
    transform = get_test_transforms()
    results = []
    missing = 0

    print("\nRunning inference on test images...")
    start_time = time.time()

    with torch.inference_mode():
        for i, rec in enumerate(test_recs):
            img_id = rec["image_id"]
            true_grade = int(rec["true_grade"])
            img_path = images_dir / f"{img_id}.png"

            if not img_path.exists():
                img_path = images_dir / f"{img_id}.jpg"
            if not img_path.exists():
                missing += 1
                continue

            # Open actual image
            image = Image.open(img_path).convert("RGB")
            tensor = transform(image).unsqueeze(0).to(device)

            # Forward pass
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
            predicted_grade = int(probs.argmax())

            results.append({
                "image_id": img_id,
                "true_grade": true_grade,
                "predicted_grade": predicted_grade,
                "score_grade_0": round(float(probs[0]), 6),
                "score_grade_1": round(float(probs[1]), 6),
                "score_grade_2": round(float(probs[2]), 6),
                "score_grade_3": round(float(probs[3]), 6),
                "score_grade_4": round(float(probs[4]), 6),
            })

            if (i + 1) % 100 == 0:
                print(f"  Processed {i+1}/{len(test_recs)} images...")

    eval_time = time.time() - start_time
    print(f"\nInference completed in {eval_time:.1f}s ({len(results)} images)")

    if missing > 0:
        print(f"WARNING: {missing} test images not found on disk!")

    # --- Save predictions CSV ---
    predictions_path = output_dir / "held_out_predictions.csv"
    fieldnames = [
        "image_id", "true_grade", "predicted_grade",
        "score_grade_0", "score_grade_1", "score_grade_2",
        "score_grade_3", "score_grade_4",
    ]
    with open(predictions_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved {len(results)} predictions to {predictions_path}")

    # --- Compute metrics exclusively from prediction CSV ---
    y_true = [r["true_grade"] for r in results]
    y_pred = [r["predicted_grade"] for r in results]

    accuracy = sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)
    qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic")
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    cm = sk_confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))

    # Argmax consistency check
    violations = 0
    for r in results:
        scores = [r[f"score_grade_{k}"] for k in range(NUM_CLASSES)]
        if int(np.argmax(scores)) != r["predicted_grade"]:
            violations += 1

    print(f"\n{'='*72}")
    print("EMPIRICAL EVALUATION RESULTS")
    print(f"{'='*72}")
    print(f"Test images evaluated: {len(results)}")
    print(f"Accuracy:   {accuracy*100:.2f}% ({sum(t==p for t,p in zip(y_true,y_pred))}/{len(y_true)})")
    print(f"QWK:        {qwk:.6f}")
    print(f"Macro F1:   {macro_f1:.6f}")
    print(f"Argmax violations: {violations}")
    print(f"\nConfusion Matrix:")
    print(f"{'':>12} Pred0  Pred1  Pred2  Pred3  Pred4  | Support")
    labels = ["No DR", "Mild", "Mod", "Severe", "PDR"]
    for i in range(NUM_CLASSES):
        row_str = "  ".join(f"{cm[i][j]:5d}" for j in range(NUM_CLASSES))
        print(f"True {labels[i]:>6}: {row_str}  | {sum(cm[i])}")

    # Per-class metrics
    print(f"\n{'Grade':<8} {'Support':>7} {'Sensitivity':>11} {'Precision':>9} {'F1':>7}")
    for c in range(NUM_CLASSES):
        support = sum(cm[c])
        tp = cm[c][c]
        fp = sum(cm[r][c] for r in range(NUM_CLASSES)) - tp
        fn = support - tp
        sens = tp / support if support > 0 else 0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1_c = 2 * prec * sens / (prec + sens) if (prec + sens) > 0 else 0
        print(f"Grade {c}  {support:7d} {sens*100:10.2f}% {prec*100:8.2f}% {f1_c:7.4f}")

    # --- Save evaluation summary ---
    eval_summary = {
        "n_test": len(results),
        "accuracy": round(accuracy, 6),
        "qwk": round(qwk, 6),
        "macro_f1": round(macro_f1, 6),
        "argmax_violations": violations,
        "confusion_matrix": cm.tolist(),
        "checkpoint_sha256": ckpt_hash,
        "checkpoint_size_mb": round(ckpt_size_mb, 2),
        "eval_time_seconds": round(eval_time, 1),
        "device": str(device),
        "pytorch_version": torch.__version__,
    }
    summary_path = output_dir / "evaluation_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(eval_summary, f, indent=2)
    print(f"\nSaved evaluation summary to {summary_path}")


if __name__ == "__main__":
    evaluate()
