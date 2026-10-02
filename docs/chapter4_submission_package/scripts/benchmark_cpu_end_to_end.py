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

    # Preferred: real APTOS images.
    !python backend/scripts/benchmark_cpu_end_to_end.py \
        --images-dir aptos2019/train_images --runs 30

    # Fallback when the 9.51 GB dataset is not available.
    !python backend/scripts/benchmark_cpu_end_to_end.py --synthetic --runs 30

Synthetic mode draws fundus-like images rather than using real ones. The five
compute stages (preprocess, forward, gradcam, compose, encode) depend only on
tensor shape and model topology, so they are exactly as valid as on real input.
The four input-handling stages (read, gate1 decode, gate2, gate3) depend on file
size and pixel statistics, so they are approximations. The script labels this in
its output and records a `compute_only_mean_ms` figure that is safe to cite
regardless. Prefer real images when you can get them.

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

# Refuse a directory whose files claim a manifest identity they do not have.
# A local directory of 640x480 placeholders named after real held-out images
# produced output that looked exactly like evidence; nothing noticed, because
# every check asked whether an image_id was in the manifest and none asked
# whether the FILE was that image. See corpus_guard.py.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from corpus_guard import assert_corpus_is_authentic  # noqa: E402

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


def generate_synthetic_fundus(width, height, seed, noise_sigma=0.6):
    """
    Draw a fundus-like image for benchmarking when the real dataset is not to hand.

    Geometry mirrors the generator the test suite already uses (circular aperture,
    optic disc, macula, branching vessels) so the validation gates behave the way
    they do on real input: Gate 2 sees plausible aperture coverage and R/B ratio,
    Gate 3 sees enough high-frequency vessel detail to clear the blur threshold.

    Gaussian noise is added on top, which is not cosmetic. Without it this image
    encodes to roughly 0.03 MB, about ninety times smaller than a real APTOS
    file, which would make the read and decode timings meaningless. The default
    sigma of 0.6 was chosen by measurement: at 2048x1536 it yields ~2.5 MB
    against the APTOS average of 2.66 MB (9.51 GB across 3,662 images).
    """
    import math
    import numpy as np
    from PIL import Image, ImageDraw

    rng = np.random.default_rng(seed)
    img = Image.new("RGB", (width, height), (5, 5, 5))
    draw = ImageDraw.Draw(img)

    # Circular aperture, jittered slightly per image
    margin = int(width * (0.06 + 0.04 * rng.random()))
    draw.ellipse([margin, margin, width - margin, height - margin], fill=(185, 65, 25))

    disc_x = int(width * (0.30 + 0.10 * rng.random()))
    disc_y = int(height * (0.45 + 0.10 * rng.random()))
    disc_r = int(width * 0.07)
    draw.ellipse([disc_x - disc_r, disc_y - disc_r, disc_x + disc_r, disc_y + disc_r],
                 fill=(240, 210, 110))

    fovea_x, fovea_y = int(width * 0.58), int(height * 0.52)
    fovea_r = int(width * 0.04)
    draw.ellipse([fovea_x - fovea_r, fovea_y - fovea_r, fovea_x + fovea_r, fovea_y + fovea_r],
                 fill=(120, 30, 15))

    # Branching vessels - the high-frequency content Gate 3 measures
    vessel = (110, 20, 15)
    branches = 14
    for i in range(branches):
        angle = (i / branches) * 2 * math.pi + rng.random() * 0.2
        x_end = int(disc_x + math.cos(angle) * (width * 0.36))
        y_end = int(disc_y + math.sin(angle) * (height * 0.36))
        draw.line([disc_x, disc_y, x_end, y_end], fill=vessel, width=max(2, width // 700))
        mx, my = (disc_x + x_end) // 2, (disc_y + y_end) // 2
        step = max(12, width // 60)
        draw.line([mx, my, mx + step, my - step], fill=vessel, width=max(1, width // 1000))
        draw.line([mx, my, mx - step, my + step], fill=vessel, width=max(1, width // 1000))

    arr = np.asarray(img).astype(np.float32)
    arr += rng.normal(0, noise_sigma, arr.shape).astype(np.float32)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="RGB")


def make_synthetic_images(count, width, height, out_dir, noise_sigma=0.6):
    """Write `count` synthetic fundus PNGs and return their paths (cached on disk)."""
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for n in range(count):
        path = os.path.join(out_dir, f"synthetic_{n:03d}.png")
        if not os.path.exists(path):
            generate_synthetic_fundus(
                width, height, seed=1000 + n, noise_sigma=noise_sigma
            ).save(path, format="PNG")
        paths.append(path)
    return paths


def resolve_images(images_dir, runs):
    """Prefer held-out test images named in the manifest; fall back to any PNG."""
    assert_corpus_is_authentic(images_dir, sample=64)

    if not os.path.isdir(images_dir):
        raise SystemExit(
            f"Image directory not found: {images_dir}\n"
            "Pass --images-dir pointing at your APTOS train_images/ directory,\n"
            "or use --synthetic to benchmark on generated fundus-like images."
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
    parser.add_argument("--synthetic", action="store_true",
                        help="Benchmark on generated fundus-like images instead of APTOS. "
                             "Compute stages stay valid; read/decode become approximate.")
    parser.add_argument("--synthetic-size", default="2048x1536",
                        help="WxH for synthetic images (default 2048x1536, APTOS-typical)")
    parser.add_argument("--synthetic-dir", default=None,
                        help="Where to write synthetic images (default: a temp directory)")
    parser.add_argument("--synthetic-noise", type=float, default=0.6,
                        help="Gaussian sigma controlling PNG entropy. Default 0.6 gives "
                             "~2.5 MB at 2048x1536, against the APTOS average of 2.66 MB.")
    args = parser.parse_args()

    if args.synthetic:
        try:
            syn_w, syn_h = (int(x) for x in args.synthetic_size.lower().split("x"))
        except ValueError:
            raise SystemExit(f"--synthetic-size must look like 2048x1536, got {args.synthetic_size!r}")

    import numpy as np
    import torch
    import torchvision.models as models
    import torchvision.transforms as transforms
    from PIL import Image

    from app.services.validation.gate1_integrity import evaluate_gate1
    from app.services.validation.gate2_relevance import evaluate_gate2
    from app.services.validation.gate3_quality import evaluate_gate3
    from app.services.ai_service import viridis_rgba_array

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
        # Call the production composition function. This stage previously held a
        # copy of the per-pixel loop, which silently kept measuring code that
        # ai_service no longer runs. Import it; never reimplement it.
        rgba = viridis_rgba_array(cam_arr)
        heatmap = Image.fromarray(rgba)
        timings["compose"].append((t() - t0) * 1000)

        t0 = t()
        buf = io.BytesIO()
        heatmap.save(buf, format="PNG")
        buf.getvalue()
        timings["encode"].append((t() - t0) * 1000)
        return True

    need = args.runs + args.warmup
    if args.synthetic:
        import tempfile
        syn_dir = args.synthetic_dir or os.path.join(
            tempfile.gettempdir(), f"dr_cdss_synth_{syn_w}x{syn_h}")
        print(f"Generating {need} synthetic fundus images at {syn_w}x{syn_h} ...")
        paths = make_synthetic_images(need, syn_w, syn_h, syn_dir, args.synthetic_noise)
        sizes = [os.path.getsize(p) for p in paths]
        mean_kb = sum(sizes) / len(sizes) / 1024
        image_source = f"synthetic ({syn_w}x{syn_h}, mean {mean_kb:.0f} KB PNG)"
    else:
        paths = resolve_images(args.images_dir, need)
        sizes = [os.path.getsize(p) for p in paths]
        mean_kb = sum(sizes) / len(sizes) / 1024
        image_source = f"APTOS ({args.images_dir}, mean {mean_kb:.0f} KB)"

    print("=" * 78)
    print("DR-CDSS END-TO-END CPU LATENCY BENCHMARK")
    print("=" * 78)
    print(f"Device     : CPU ({platform.processor() or platform.machine()})")
    print(f"PyTorch    : {torch.__version__}   threads={torch.get_num_threads()}")
    print(f"Checkpoint : {os.path.basename(checkpoint)}")
    print(f"Images     : {image_source}")
    print(f"Warm-up    : {args.warmup}    Measured: {args.runs}")
    if args.synthetic:
        print("-" * 78)
        print("SYNTHETIC MODE - scope of what these numbers support:")
        print("  VALID    : preprocess, forward, gradcam, compose, encode.")
        print("             These depend on tensor shape and model topology, which are")
        print("             identical to a real request, not on image content.")
        print("  APPROX   : read, gate1 (decode), gate2, gate3. These depend on file size")
        print("             and pixel statistics. Noise is added to keep PNG entropy in a")
        print("             realistic range, but decode cost is an estimate, not a")
        print("             measurement. Compare the mean KB above against your own APTOS")
        print("             files to judge how close it lands.")
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
        hints = {
            "gate2": "Gate 2 computes aperture coverage and channel ratios over the "
                     "FULL-resolution image; downsampling first would cut this sharply.",
            "gate3": "Gate 3 computes Laplacian variance and illumination over the "
                     "FULL-resolution image; lowering its resize ceiling would help.",
            "compose": "Heatmap composition. If this is large, confirm it is calling "
                       "viridis_rgba_array and not a per-pixel loop.",
            "encode": "PNG serialisation of the 512x512 overlay; a lower compress_level "
                      "trades file size for speed.",
            "forward": "Model forward pass - this is the irreducible floor.",
        }
        if worst["stage"] in hints:
            print(f"Note: {hints[worst['stage']]}")

        gates = sum(r["mean_ms"] for r in rows if r["stage"] in ("gate2", "gate3"))
        if gates > 0:
            print(f"Validation gates 2+3 together: {gates:.1f} ms "
                  f"({100 * gates / mean_total:.0f}% of the request).")

    compute_stages = ["preprocess", "forward", "gradcam", "compose", "encode"]
    compute_total = sum(r["mean_ms"] for r in rows if r["stage"] in compute_stages)
    if args.synthetic:
        print(f"\nCompute-only subtotal (content-independent, fully valid on synthetic "
              f"input): {compute_total:.1f} ms")
        print("Cite that figure rather than the total if you report synthetic results.")

    summary = {
        "scope": "end-to-end request path (read -> gates -> preprocess -> forward -> Grad-CAM -> compose -> encode)",
        "image_source": "synthetic" if args.synthetic else "aptos",
        "image_source_detail": image_source,
        "mean_image_bytes": int(sum(sizes) / len(sizes)),
        "synthetic_caveat": (
            "read/gate1/gate2/gate3 are approximations on synthetic input; "
            "preprocess/forward/gradcam/compose/encode are content-independent and valid."
            if args.synthetic else None
        ),
        "compute_only_mean_ms": round(compute_total, 3),
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
