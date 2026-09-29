# Computational Efficiency & System Resource Benchmark

## Metadata & Academic Context
- **Research Project:** AI-Based Clinical Decision Support System for Early Detection of Diabetic Retinopathy
- **Researcher:** Onyekelu Chukwuebuka Elochukwu (2024516020FN)
- **Programme:** PGD Computer Science, Faculty of Physical Sciences
- **Benchmark Objective:** Objective i supporting evidence (inference latency & efficiency)
- **Execution Date:** 2026-09-29
- **Raw Evidence:** [`cpu_end_to_end_benchmark.json`](cpu_end_to_end_benchmark.json), [`cpu_end_to_end_benchmark.csv`](cpu_end_to_end_benchmark.csv), [`benchmark_timings.csv`](benchmark_timings.csv), [`benchmark_summary.json`](benchmark_summary.json)

---

## 1. End-to-End CPU Latency (the deployment figure)

This is what a clinician waits for: the complete request path, on the CPU the system actually deploys to, measured over 30 real held-out APTOS images after 3 warm-up requests.

**Environment:** x86_64 CPU, 4 threads, PyTorch 2.11.0+cpu, no GPU — the harness clears `CUDA_VISIBLE_DEVICES` before importing torch, so it cannot silently measure an accelerator. Input images averaged 1,807 KB.

| Measure | Value |
| :--- | :---: |
| **Mean end-to-end latency** | **315.25 ms** |
| **Median (P50)** | **227.77 ms** |
| **95th percentile (P95)** | **624.58 ms** |
| Minimum | 151.86 ms |
| Maximum | 631.75 ms |

Sub-second at the 95th percentile on a commodity 4-thread CPU with no accelerator. For an assisted-review workflow — a clinician uploads a fundus photograph, then reads the result — this is comfortably interactive.

### Stage breakdown

| Stage | Mean ms | Median | P95 | Min | Max | % of request |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `read` | 1.64 | 0.63 | 3.55 | 0.21 | 22.95 | 0.5% |
| `gate1` — integrity & decode | 6.43 | 3.95 | 19.35 | 1.33 | 19.86 | 2.0% |
| **`gate2` — retinal relevance** | **118.35** | 61.62 | 318.02 | 25.01 | 318.59 | **37.5%** |
| **`gate3` — technical quality** | **68.07** | 40.40 | 156.88 | 10.45 | 157.33 | **21.6%** |
| `preprocess` | 15.12 | 8.76 | 27.68 | 4.91 | 34.12 | 4.8% |
| `forward` — EfficientNet-B0 | 33.48 | 33.52 | 39.54 | 28.94 | 42.50 | 10.6% |
| `gradcam` — gradients & CAM | 0.59 | 0.56 | 0.71 | 0.54 | 0.75 | 0.2% |
| `compose` — heatmap render | 28.86 | 27.39 | 34.61 | 25.73 | 34.63 | 9.2% |
| `encode` — PNG serialisation | 41.48 | 40.55 | 49.47 | 35.26 | 50.40 | 13.2% |
| **TOTAL** | **315.25** | 227.77 | 624.58 | 151.86 | 631.75 | 100% |

The stage means sum to 314.02 ms against a measured total of 315.25 ms. The 1.24 ms difference (0.4%) is work between the timed regions — registering and removing the Grad-CAM forward hook, the argmax, tensor indexing — which is attributed to no stage. `TOTAL` is wall-clock around the whole request and is the figure to cite; the stage rows account for 99.6% of it.

Every value in this section is reproduced verbatim from [`cpu_end_to_end_benchmark.json`](cpu_end_to_end_benchmark.json), rounded to two decimals.

---

## 2. What the Breakdown Shows

**Inference is not the bottleneck.** The model forward pass is **33.48 ms — 10.6%** of the request. Image *validation* costs nearly six times that.

**Validation dominates.** Gates 2 and 3 together account for **186.42 ms, 59.1% of the request**. Both compute NumPy statistics — aperture coverage, channel ratios, Laplacian variance, illumination — over the **full-resolution** image, before anything is downsampled. Gate 3 already shrinks very large inputs; Gate 2 does not.

Grouped by kind:

| Category | Mean ms | Share |
| :--- | ---: | ---: |
| Input handling (`read`, `gate1`, `gate2`, `gate3`) | 194.49 | **61.7%** |
| Model & explainability (`preprocess`, `forward`, `gradcam`, `compose`, `encode`) | 119.53 | 37.9% |

**Latency scales with input resolution, not with disease severity.** Mean exceeds median by a factor of 1.38 (315 vs 228 ms), and the spread is widest exactly where it would be if resolution drove cost: `gate2` has a median of 61.62 ms against a P95 of 318.02 ms, a 5× range. APTOS image dimensions vary substantially and the full-resolution gates pay for every extra pixel. The model stages, operating on a fixed 224×224 tensor, are correspondingly stable — `forward` spans only 28.94–42.50 ms across all 30 requests.

**This identified the next optimisation, which has since been applied.** See §3.3. The §1 table above remains the *pre-optimisation* baseline, kept because it is the measurement that motivated the change and because the post-optimisation end-to-end figure has not yet been measured.

---

## 3. Corrections Applied to Reach This Measurement

Two defects were found while producing this figure. Both are recorded because each affected a previously published number.

**Heatmap composition ran a per-pixel Python loop.** Grad-CAM overlay rendering built its 512×512 RGBA output with a nested loop calling the colormap function once per pixel — 262,144 interpreter iterations per request. Now vectorised, with byte-identical output verified by 13 tests:

| | Before | After |
| :--- | ---: | ---: |
| `compose` (512×512 overlay) | 307.87 ms | **28.86 ms** (10.7× faster) |

Before that fix, this same harness measured a **592.02 ms** end-to-end total with composition alone at 52% of the request.

**The harness initially measured a stale copy of that loop.** It held its own inline copy of the composition code, written before the vectorisation and not updated alongside it, so its first run reported timings for code the application no longer executed. It now imports `viridis_rgba_array`, the same function the inference service calls. The 592.02 ms figure is withdrawn; §1 comes from the corrected harness.

### 3.3 Validation gates now analyse a subsample

The §1 breakdown showed gates 2 and 3 costing 186.42 ms, 59.1% of the request, because both computed their pixel statistics over the full-resolution array.

Those statistics — aperture coverage, channel means, contrast standard deviation, the proportion of extreme pixels — describe the pixel *distribution*, not any individual pixel, so a large uniform sample estimates them just as well. Both gates now compute them on a copy whose longest side is `VALIDATION_ANALYSIS_MAX_DIM` (default 512), sampled with **nearest-neighbour**.

Nearest is a correctness choice, not a speed one. It samples pixels where bilinear and area resampling average them, and averaging would (a) smooth the aperture boundary into intermediate luminances that cross the foreground threshold, and (b) pull extreme values toward the mean — biasing exactly the statistics Gate 3 uses to detect washout and underexposure, in the unsafe direction. Measured across a threshold-spanning corpus, bilinear moved aperture coverage by up to 0.011 where nearest moved it by 0.0007.

Measured on a 2048×1536 image:

| | Full resolution | Subsampled | |
| :--- | ---: | ---: | :--- |
| `gate2` | 309.7 ms | **59.5 ms** | 5.2× |
| `gate3` | 393.8 ms | **205.6 ms** | 1.9× |
| combined | 703.4 ms | **265.0 ms** | **2.7×** |

Gate 3 improves less because its Laplacian variance is deliberately excluded. Sharpness is a spatial derivative, so its value depends on resolution by definition and its 60.0 threshold is calibrated against its own existing 1024px path. That path is untouched, and a test asserts the variance does not move when the analysis resolution changes.

**Decision preservation.** 38 unit tests assert that the accept/reject verdict is identical to full resolution across images placed deliberately near each threshold. Because those images are synthetic and the claim concerns real clinical images, [`backend/scripts/verify_gate_downsampling.py`](../../backend/scripts/verify_gate_downsampling.py) runs the same comparison over a real APTOS directory and fails on any changed verdict:

```bash
python backend/scripts/verify_gate_downsampling.py aptos2019/train_images
```

**Not yet claimed:** the post-optimisation end-to-end total. §1 is the pre-optimisation baseline. Re-running the end-to-end harness is required before any improved figure is stated here.

---

## 4. Model Efficiency (device-independent)

| Property | Value |
| :--- | :---: |
| Model weights file size | **15.60 MB** (16,358,249 bytes) |
| Total parameters | 4,013,953 |
| Trainable parameters at inference | **0** (frozen, `eval()`, `requires_grad=False`) |
| Theoretical FLOPs @ 224² | ~0.39 GFLOPs |

These hold regardless of hardware: the checkpoint ships in a container layer without special handling, the compute cost sits at the bottom of the modern CNN range, and the inference graph is fully frozen, so any prediction is attributable to a fixed, hash-verified artefact.

---

## 5. Training-Environment Reference: GPU Forward Pass

Retained for comparison, **not** as a deployment figure. Measured on the Google Colab Tesla T4 used for training, timing the forward pass alone with CUDA events (10 warm-up, 100 measured passes, batch size 1):

| Measure | Value |
| :--- | :---: |
| Mean | 8.36 ms |
| Median | 8.34 ms |
| P95 | 8.86 ms |
| Min / Max | 7.91 / 9.11 ms |

Raw data in [`benchmark_timings.csv`](benchmark_timings.csv). The corresponding CPU forward pass is 33.48 ms — a 4× gap that is unsurprising and, as §2 shows, largely irrelevant to end-to-end response time.

---

## 6. Reproduction

```bash
# End-to-end on CPU against real APTOS images (the §1 table)
python backend/scripts/benchmark_cpu_end_to_end.py \
    --images-dir aptos2019/train_images --runs 30

# Forward pass only (the §5 table); selects CUDA when present, else CPU
python backend/scripts/benchmark_resources.py
```

`benchmark_cpu_end_to_end.py` draws its images from the held-out split named in [`dataset_split_manifest.csv`](dataset_split_manifest.csv), so it profiles the same cohort the model was evaluated on. Beyond the backend dependencies it needs only `pydantic-settings`; the validation gates require no database.

Where the 9.51 GB dataset is unavailable, `--synthetic` generates calibrated fundus-like images, with noise tuned so PNG entropy lands within ~2% of real files. On synthetic input the five compute stages remain exactly valid — they depend only on tensor shape and model topology — while `read`, `gate1`, `gate2` and `gate3` become approximations. The script labels this and reports a `compute_only_mean_ms` figure that is safe to cite either way. **The §1 table is from real images, not synthetic.**
