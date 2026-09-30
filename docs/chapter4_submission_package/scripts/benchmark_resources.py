#!/usr/bin/env python3
"""
EfficientNet-B0 Inference Benchmark — Real Forward Pass Timing
PGD Computer Science, Faculty of Physical Sciences

Loads the actual trained checkpoint, opens a real test image,
and times 100 individual forward passes. Records every measurement
to CSV along with system metadata.
"""

import os
import csv
import sys
import json
import time
import platform
import hashlib
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

NUM_CLASSES = 5
IMAGE_SIZE = 224
WARMUP_PASSES = 10
MEASUREMENT_PASSES = 100


def get_transforms():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def benchmark():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 72)
    print("PGD Computer Science — EfficientNet-B0 Inference Benchmark")
    print(f"Device: {device}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Python: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print(f"Processor: {platform.processor()}")
    print(f"CPU threads: {os.cpu_count()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 72)

    # --- Paths ---
    root = Path(__file__).resolve().parents[2]
    checkpoint_path = root / "backend" / "models" / "weights" / "efficientnet_b0_dr.pth"
    manifest_path = root / "docs" / "chapter4" / "dataset_split_manifest.csv"
    images_dir = root / "storage" / "datasets" / "aptos2019" / "train_images"
    output_dir = root / "docs" / "chapter4"
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load model ---
    print("\nLoading checkpoint...")
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=False),
        nn.Linear(in_features, NUM_CLASSES),
    )
    state_dict = torch.load(str(checkpoint_path), map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print("Model loaded and set to eval mode.")

    # --- Find a real test image ---
    with open(manifest_path, "r", encoding="utf-8") as f:
        records = list(csv.DictReader(f))
    test_recs = [r for r in records if r["split"] == "test"]

    test_image_path = None
    for rec in test_recs:
        p = images_dir / f"{rec['image_id']}.png"
        if p.exists():
            test_image_path = p
            break
        p = images_dir / f"{rec['image_id']}.jpg"
        if p.exists():
            test_image_path = p
            break

    if test_image_path is None:
        raise FileNotFoundError("No test images found on disk. Download APTOS 2019 dataset first.")

    print(f"Benchmark image: {test_image_path}")
    print(f"Image size on disk: {test_image_path.stat().st_size:,} bytes")

    # --- Prepare input tensor ---
    transform = get_transforms()
    image = Image.open(test_image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    # --- Warmup ---
    print(f"\nWarm-up: {WARMUP_PASSES} passes...")
    with torch.inference_mode():
        for _ in range(WARMUP_PASSES):
            _ = model(input_tensor)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # --- Timed measurement ---
    print(f"Measurement: {MEASUREMENT_PASSES} passes...")
    timings = []

    with torch.inference_mode():
        for run_idx in range(1, MEASUREMENT_PASSES + 1):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                _ = model(input_tensor)
                end.record()
                torch.cuda.synchronize()
                elapsed_ms = start.elapsed_time(end)
            else:
                start = time.perf_counter_ns()
                _ = model(input_tensor)
                elapsed_ns = time.perf_counter_ns() - start
                elapsed_ms = elapsed_ns / 1_000_000.0

            timings.append({
                "run": run_idx,
                "inference_time_ms": round(elapsed_ms, 4),
            })

    # --- Statistics ---
    times_ms = [t["inference_time_ms"] for t in timings]
    times_ms_sorted = sorted(times_ms)
    mean_ms = sum(times_ms) / len(times_ms)
    median_ms = times_ms_sorted[len(times_ms_sorted) // 2]
    p95_ms = times_ms_sorted[int(0.95 * len(times_ms_sorted))]
    p99_ms = times_ms_sorted[int(0.99 * len(times_ms_sorted))]
    min_ms = min(times_ms)
    max_ms = max(times_ms)

    print(f"\n{'='*72}")
    print("BENCHMARK RESULTS")
    print(f"{'='*72}")
    print(f"Runs:   {MEASUREMENT_PASSES}")
    print(f"Mean:   {mean_ms:.2f} ms = {mean_ms/1000:.5f} s")
    print(f"Median: {median_ms:.2f} ms")
    print(f"P95:    {p95_ms:.2f} ms")
    print(f"P99:    {p99_ms:.2f} ms")
    print(f"Min:    {min_ms:.2f} ms")
    print(f"Max:    {max_ms:.2f} ms")
    print(f"Threshold (250 ms): {'PASS' if mean_ms < 250 else 'FAIL'}")

    # --- Save timings CSV ---
    timings_path = output_dir / "benchmark_timings.csv"
    with open(timings_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["run", "inference_time_ms"])
        writer.writeheader()
        writer.writerows(timings)
    print(f"\nSaved {MEASUREMENT_PASSES} timings to {timings_path}")

    # --- Save benchmark summary ---
    summary = {
        "measurement_scope": "model forward pass only (inference_mode, batch_size=1)",
        "warmup_passes": WARMUP_PASSES,
        "measurement_passes": MEASUREMENT_PASSES,
        "mean_ms": round(mean_ms, 4),
        "median_ms": round(median_ms, 4),
        "p95_ms": round(p95_ms, 4),
        "p99_ms": round(p99_ms, 4),
        "min_ms": round(min_ms, 4),
        "max_ms": round(max_ms, 4),
        "threshold_ms": 250.0,
        "threshold_pass": mean_ms < 250,
        "device": str(device),
        "pytorch_version": torch.__version__,
        "python_version": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        "benchmark_image": str(test_image_path),
        "timestamp": datetime.now().isoformat(),
    }
    summary_path = output_dir / "benchmark_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved benchmark summary to {summary_path}")


if __name__ == "__main__":
    benchmark()
