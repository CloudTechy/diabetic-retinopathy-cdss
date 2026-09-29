#!/usr/bin/env python3
"""
End-to-end CPU latency benchmark for the DR-CDSS inference path.

This measures what a clinician actually waits for, on the hardware the system
actually deploys to. It is deliberately different from benchmark_resources.py,
which times the model forward pass alone on whatever accelerator is present.

Stages timed separately, in the order a real request executes them:

    1. read        - read the image file from disk into bytes
    2. gate1       - Gate 1: file integrity, magic bytes, decode, dimensions
    3. gate2       - Gate 2: retinal relevance (aperture coverage, R/B ratio)
    4. gate3       - Gate 3: technical quality (Laplacian blur, illumination)
    5. preprocess  - RGB convert, resize to 224, ToTensor, ImageNet normalise
    6. forward     - EfficientNet-B0 forward pass with the Grad-CAM hook
    7. gradcam     - backward gradients on features.8, CAM computation
    8. compose     - CAM -> 512x512 RGBA heatmap
    9. encode      - PNG serialisation of the heatmap

Reporting per stage, rather than one total, is the point: a single number tells
you the request is slow but not which part to fix.

Usage (Colab, CPU runtime):

    !pip install -q pydantic-settings
    %cd /content/diabetic-retinopathy-cdss
    !python backend/scripts/benchmark_cpu_end_to_end.py --runs 30

    # against your own image directory:
    !python backend/scripts/benchmark_cpu_end_to_end.py \
        --images-dir aptos2019/train_images --runs 30

Outputs docs/chapter4/cpu_end_to_end_benchmark.{json,csv}.
"""

import argparse
import csv
import io
import json
import os
import platform
import statistics
import sys
import time

# Force CPU before torch is imported anywhere. The deployment target is CPU;
# benchmarking on an available GPU would reproduce the exact error this script
# exists to correct.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("MODEL_DEVICE", "cpu")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BACKEND_ROOT = os.path.join(REPO_ROOT, "backend")
if BACKEND_ROOT not in sys.path:
    sys.path.insert(0, BACKEND_ROOT)

CHAPTER4 = os.path.join(REPO_ROOT, "docs", "chapter4")
MANIFEST = os.path.join(CHAPTER4, "dataset_split_manifest.csv")

STAGES = ["read", "gate1", "gate2", "gate3", "preprocess",
          "forward", "gradcam", "compose", "encode"]


def percentile(values, pct):
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(int(pct / 100.0 * len(ordered)), len(ordered) - 1)
    return ordered[idx]


def resolve_images(images_dir, runs):
    """Prefer held-out test images named in the manifest; fall back to any PNG."""
    if not os.path.isdir(images_dir):
        raise SystemExit(
            f"Image directory not found: {images_dir}\n"
            "Pass --images-dir pointing at your APTOS train_images/ directory."
        )

    wanted = []
    if os.path.exists(MANIFEST):
        with open(MANIFEST, newline="", encoding="utf-8") as fh:
            wanted = [r["image_id"] for r in csv.DictReader(fh) if r["split"] == "test"]

    paths = []
    for img_id in wanted:
        candidate = os.path.join(images_dir, f"{img_id}.png")
        if os.path.exists(candidate):
            paths.append(candidate)
        if len(paths) >= runs:
            break

    if not paths:
        paths = [
            os.path.join(images_dir, f)
            for f in sorted(os.listdir(images_dir))
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ][:runs]

    if not paths:
        raise SystemExit(f"No usable images found in {images_dir}")

    # Cycle if the caller asked for more runs than available images.
    while len(paths) < runs:
        paths.append(paths[len(paths) % max(len(paths), 1)])
    return paths[:runs]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images-dir", default="aptos2019/train_images",
                        help="Directory holding the APTOS images")
    parser.add_argument("--runs", type=int, default=30, help="Measured requests (default 30)")
    parser.add_argument("--warmup", type=int, default=3, help="Warm-up requests (default 3)")
    parser.add_argument("--checkpoint", default=None, help="Override checkpoint path")
    args = parser.parse_args()

    import numpy as np
    import torch
    import torchvision.models as models
    import torchvision.transforms as transforms
    from PIL import Image

    from app.services.validation.gate1_integrity import evaluate_gate1
    from app.services.validation.gate2_relevance import evaluate_gate2
    from app.services.validation.gate3_quality import evaluate_gate3
    from app.services.ai_service import generate_viridis_colormap

    torch.set_grad_enabled(True)   # Grad-CAM needs gradients
    device = torch.device("cpu")

    checkpoint = args.checkpoint or os.path.join(
        REPO_ROOT, "backend", "models", "weights", "efficientnet_b0_dr.pth")
    if not os.path.exists(checkpoint):
        raise SystemExit(f"Checkpoint not found: {checkpoint}")

    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.2), torch.nn.Linear(in_features, 5))
    model.load_state_dict(
        torch.load(checkpoint, map_location=device, weights_only=True), strict=True)
    model.eval().to(device)

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    def one_request(path, timings):
        t = time.perf_counter

        t0 = t()
        with open(path, "rb") as fh:
            image_bytes = fh.read()
        timings["read"].append((t() - t0) * 1000)

        t0 = t()
        g1, pil_image = evaluate_gate1(image_bytes, os.path.basename(path))
        timings["gate1"].append((t() - t0) * 1000)
        if pil_image is None:
            return False

        t0 = t()
        evaluate_gate2(pil_image)
        timings["gate2"].append((t() - t0) * 1000)

        t0 = t()
        evaluate_gate3(pil_image)
        timings["gate3"].append((t() - t0) * 1000)

        t0 = t()
        tensor = preprocess(pil_image.convert("RGB")).unsqueeze(0).to(device)
        tensor.requires_grad = True
        timings["preprocess"].append((t() - t0) * 1000)

        activations = []
        handle = model.features[8].register_forward_hook(
            lambda m, i, o: activations.append(o))

        t0 = t()
        logits = model(tensor)
        handle.remove()
        grade = int(torch.argmax(logits, dim=1).item())
        timings["forward"].append((t() - t0) * 1000)

        t0 = t()
        grads = torch.autograd.grad(logits[0, grade], activations[0])[0]
        weights = torch.mean(grads, dim=(2, 3), keepdim=True)
        cam = torch.relu(torch.sum(weights * activations[0], dim=1)).squeeze().detach().numpy()
        cam = cam / np.max(cam) if np.max(cam) > 0 else np.zeros_like(cam)
        timings["gradcam"].append((t() - t0) * 1000)

        t0 = t()
        cam_pil = Image.fromarray((cam * 255).astype(np.uint8)).resize(
            (512, 512), Image.Resampling.BILINEAR)
        cam_arr = np.array(cam_pil, dtype=np.float32) / 255.0
        rgba = np.zeros((512, 512, 4), dtype=np.uint8)
        for i in range(512):
            for j in range(512):
                rgba[i, j] = generate_viridis_colormap(float(cam_arr[i, j]))
        heatmap = Image.fromarray(rgba, mode="RGBA")
        timings["compose"].append((t() - t0) * 1000)

        t0 = t()
        buf = io.BytesIO()
        heatmap.save(buf, format="PNG")
        buf.getvalue()
        timings["encode"].append((t() - t0) * 1000)
        return True

    paths = resolve_images(args.images_dir, args.runs + args.warmup)

    print("=" * 78)
    print("DR-CDSS END-TO-END CPU LATENCY BENCHMARK")
    print("=" * 78)
    print(f"Device     : CPU ({platform.processor() or platform.machine()})")
    print(f"PyTorch    : {torch.__version__}   threads={torch.get_num_threads()}")
    print(f"Checkpoint : {os.path.basename(checkpoint)}")
    print(f"Warm-up    : {args.warmup}    Measured: {args.runs}")
    print("-" * 78)

    scratch = {s: [] for s in STAGES}
    for p in paths[:args.warmup]:
        one_request(p, scratch)
    print(f"Warm-up complete ({args.warmup} requests).")

    timings = {s: [] for s in STAGES}
    totals = []
    completed = 0
    for n, p in enumerate(paths[args.warmup:args.warmup + args.runs], 1):
        before = {s: len(timings[s]) for s in STAGES}
        t0 = time.perf_counter()
        ok = one_request(p, timings)
        total_ms = (time.perf_counter() - t0) * 1000
        if not ok:
            for s in STAGES:
                del timings[s][before[s]:]
            continue
        totals.append(total_ms)
        completed += 1
        if n % 10 == 0:
            print(f"  {n}/{args.runs} ...")

    if not totals:
        raise SystemExit("No request completed; every image failed Gate 1.")

    print("-" * 78)
    print(f"{'Stage':<12}{'Mean ms':>10}{'Median':>10}{'P95':>10}{'Min':>9}{'Max':>9}{'% total':>9}")
    print("-" * 78)

    mean_total = statistics.mean(totals)
    rows = []
    for s in STAGES:
        v = timings[s]
        if not v:
            continue
        m = statistics.mean(v)
        rows.append({
            "stage": s,
            "mean_ms": round(m, 3),
            "median_ms": round(statistics.median(v), 3),
            "p95_ms": round(percentile(v, 95), 3),
            "min_ms": round(min(v), 3),
            "max_ms": round(max(v), 3),
            "pct_of_total": round(100 * m / mean_total, 1),
        })
        print(f"{s:<12}{m:>10.2f}{statistics.median(v):>10.2f}"
              f"{percentile(v, 95):>10.2f}{min(v):>9.2f}{max(v):>9.2f}"
              f"{100 * m / mean_total:>8.1f}%")

    print("-" * 78)
    print(f"{'TOTAL':<12}{mean_total:>10.2f}{statistics.median(totals):>10.2f}"
          f"{percentile(totals, 95):>10.2f}{min(totals):>9.2f}{max(totals):>9.2f}{100.0:>8.1f}%")
    print("=" * 78)

    if rows:
        worst = max(rows, key=lambda r: r["mean_ms"])
        print(f"\nDominant stage: '{worst['stage']}' at {worst['mean_ms']:.1f} ms "
              f"({worst['pct_of_total']}% of the request).")
        if worst["stage"] == "compose":
            print("Note: heatmap composition runs a 512x512 nested Python loop "
                  "(262,144 iterations) calling generate_viridis_colormap per pixel.\n"
                  "      Vectorising it with numpy would remove most of this cost.")

    summary = {
        "scope": "end-to-end request path (read -> gates -> preprocess -> forward -> Grad-CAM -> compose -> encode)",
        "device": "cpu",
        "cpu": platform.processor() or platform.machine(),
        "torch_threads": torch.get_num_threads(),
        "pytorch": torch.__version__,
        "python": platform.python_version(),
        "warmup": args.warmup,
        "runs_requested": args.runs,
        "runs_completed": completed,
        "total_mean_ms": round(mean_total, 3),
        "total_median_ms": round(statistics.median(totals), 3),
        "total_p95_ms": round(percentile(totals, 95), 3),
        "total_min_ms": round(min(totals), 3),
        "total_max_ms": round(max(totals), 3),
        "stages": rows,
    }

    os.makedirs(CHAPTER4, exist_ok=True)
    json_path = os.path.join(CHAPTER4, "cpu_end_to_end_benchmark.json")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    csv_path = os.path.join(CHAPTER4, "cpu_end_to_end_benchmark.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWritten: {json_path}")
    print(f"Written: {csv_path}")


if __name__ == "__main__":
    main()
