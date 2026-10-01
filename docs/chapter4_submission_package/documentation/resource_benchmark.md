# Computational Efficiency & System Resource Benchmark

## Metadata & Academic Context
- **Research Project:** AI-Based Clinical Decision Support System for Early Detection of Diabetic Retinopathy
- **Researcher:** Onyekelu Chukwuebuka Elochukwu (2024516020FN)
- **Programme:** PGD Computer Science, Faculty of Physical Sciences
- **Benchmark Objective:** Objective i supporting evidence (inference latency & efficiency)
- **Execution Date:** 2026-10-01 (§1, §2 — calibrated admission thresholds); 2026-09-29 (§1a baseline comparison, §3 corrections)
- **Raw Evidence:** [`cpu_end_to_end_benchmark.json`](cpu_end_to_end_benchmark.json), [`cpu_end_to_end_benchmark.csv`](cpu_end_to_end_benchmark.csv), [`benchmark_timings.csv`](benchmark_timings.csv), [`benchmark_summary.json`](benchmark_summary.json)

---

## 1. End-to-End CPU Latency (the deployment figure)

The complete request path, on the CPU the system deploys to, over 30 real held-out APTOS images after 3 warm-up requests.

**Environment:** x86_64 CPU, 4 threads, PyTorch 2.11.0+cpu, Python 3.13.15, no GPU — the harness clears `CUDA_VISIBLE_DEVICES` before importing torch, so it cannot silently measure an accelerator. Input images averaged 1,798 KB.

| Measure | Value |
| :--- | :---: |
| **Mean end-to-end latency** | **254.31 ms** |
| **Median (P50)** | **213.30 ms** |
| **95th percentile (P95)** | **447.95 ms** |
| Minimum | 163.37 ms |
| Maximum | 454.47 ms |

Comfortably interactive for an assisted-review workflow on a commodity 4-thread CPU with no accelerator.

> [!NOTE]
> **This run is not comparable in absolute milliseconds to the one in §1a.** It
> was measured in a later Colab session, and §1a documents the measured spread
> between sessions on identical code: 1.14× on the most stable stage, 3.14× on
> the I/O-bound ones. Every image also now passes the admission gates and
> executes the full path, where the earlier thresholds rejected the entire
> corpus at Gate 3.
>
> What survives between sessions is the stage **share**, because it is internal
> to a single run. Cite the shares for the argument and the milliseconds only
> as an order of magnitude.

### Stage breakdown

| Stage | Mean ms | Median | P95 | Min | Max | % of request |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `read` | 12.91 | 9.81 | 40.01 | 0.68 | 42.09 | 5.1% |
| `gate1` — integrity & decode | 6.74 | 4.14 | 20.09 | 1.51 | 20.72 | 2.7% |
| **`gate2` — retinal relevance** | **78.23** | 46.16 | 202.76 | 17.72 | 206.23 | **30.8%** |
| `gate3` — technical quality | 28.33 | 26.83 | 41.36 | 11.01 | 41.57 | 11.1% |
| `preprocess` | 14.44 | 9.12 | 29.24 | 5.25 | 29.76 | 5.7% |
| `forward` — EfficientNet-B0 | 36.39 | 35.54 | 43.26 | 32.29 | 50.50 | 14.3% |
| `gradcam` — gradients & CAM | 0.68 | 0.66 | 0.85 | 0.60 | 0.94 | 0.3% |
| `compose` — heatmap render | 31.17 | 29.30 | 38.01 | 27.20 | 38.95 | 12.3% |
| **`encode` — PNG serialisation** | **43.84** | 43.59 | 49.73 | 37.08 | 52.34 | **17.2%** |
| **TOTAL** | **254.31** | 213.30 | 447.95 | 163.37 | 454.47 | 100% |

Stage means sum to 252.71 ms against a measured total of 254.31 ms. The 1.60 ms difference (0.6%) is work between the timed regions — Grad-CAM hook registration and removal, the argmax, tensor indexing — belonging to no stage. `TOTAL` is wall-clock around the whole request and is the figure to cite.

Every value here is reproduced verbatim from [`cpu_end_to_end_benchmark.json`](cpu_end_to_end_benchmark.json), rounded to two decimals.

---

## 1a. Comparison With the Pre-Optimisation Baseline — read the caveat

> [!IMPORTANT]
> Every figure in this section comes from the **2026-09-29 pair of runs**, before
> and after the gate optimisation. The "Post-optimisation" column is *not* §1's
> run: §1 was measured later, on a different session, at calibrated thresholds.
> These numbers are kept verbatim because the argument below depends on the two
> columns having been measured against each other.

An earlier run of this harness, before the validation gates were optimised, measured **315.25 ms** mean against the post-optimisation **212.54 ms**. Comparing the two naively gives 1.91×, and that number would be misleading.

**The two runs were not on the same machine.** Colab allocates different CPUs between sessions. The seven stages whose code did not change between the runs were themselves faster the second time:

| Reference set | Baseline (pre-optimisation) | Post-optimisation (2026-09-29) | Apparent factor |
| :--- | ---: | ---: | ---: |
| `forward` alone (fixed 224² tensor, no I/O — most stable) | 33.48 ms | 29.24 ms | **1.14×** |
| Fixed-size pure compute (`forward`, `gradcam`, `compose`) | 62.92 ms | 47.74 ms | 1.32× |
| All seven unchanged stages | 127.59 ms | 91.02 ms | 1.40× |
| `read` + `gate1` (disk-cache dominated) | 8.07 ms | 2.57 ms | 3.14× |

The spread — 1.14× to 3.14× — shows the difference is not a single scalar, and that the I/O-bound stages benefited most from a warmer page cache.

**Honest bounds.** Taking 1.14×–1.40× as the plausible machine factor, the post-optimisation run normalised back onto the baseline hardware lands between **189 ms and 231 ms**, giving an attributable end-to-end improvement of roughly **1.4× to 1.7×** rather than 1.91×.

For gates 2 and 3 specifically:

| | Baseline | Post-optimisation (2026-09-29) | Raw | Machine-normalised |
| :--- | ---: | ---: | ---: | ---: |
| `gate2` + `gate3` | 186.42 ms | **72.42 ms** | 2.57× | **~1.8×–2.3×** |
| Share of the request | 59.1% | **43.9%** | | |

**What is unambiguous**, because it is internal to a single run and therefore immune to hardware variation: validation fell from **59.1%** of the request to **43.9%**, and `gate3` fell from 21.6% to 9.9%.

### Gate 2 is still the largest stage, and the reason is instructive

At **78.23 ms it remains 30.8%** of the request in §1 (56.06 ms, 34.0% in the 2026-09-29 run), far more than the 5.2× local speedup predicted. The residual is not the statistics — those now run on a 512px subsample — it is **the downsampling itself**. Reducing a 3216×2136 image requires reading every source pixel once, and `rgb_image = pil_image.convert("RGB")` runs at full resolution before that.

Gates 1, 2 and 3 each decode or convert the full-resolution image independently. Decoding once and sharing a single reduced copy across all three is the next available gain, and it is **not** implemented. `gate2` also retains the widest spread in the run (32.33 ms median against a 142.99 ms P95), consistent with cost tracking source resolution.

---

## 2. What the Breakdown Shows

**Inference is not the bottleneck.** The model forward pass is **36.39 ms — 14.3%** of the request. Input handling costs three and a half times that.

**Validation leads, though by much less than before the optimisation.** Gates 2 and 3 account for **106.55 ms, 41.9%** of the request, against 59.1% before it.

Grouped by kind:

| Category | Mean ms | Share |
| :--- | ---: | ---: |
| Input handling (`read`, `gate1`, `gate2`, `gate3`) | 126.20 | **49.6%** |
| Model & explainability (`preprocess`, `forward`, `gradcam`, `compose`, `encode`) | 126.51 | 49.7% |

The two halves are now almost exactly balanced. **PNG serialisation of the Grad-CAM overlay (`encode`, 43.84 ms, 17.2%) is the second-largest stage** and costs more than the forward pass. A lower `compress_level`, or emitting the overlay at the resolution the viewer actually blends it at, would reduce it; neither is implemented.

**Latency scales with input resolution, not with disease severity.** Mean exceeds median by a factor of 1.19 (254.31 vs 213.30 ms), and the spread is widest where resolution drives cost: `gate2` runs a 46.16 ms median against a 202.76 ms P95, a 4.4× range, because the reduction step must still read every source pixel. The model stages, operating on a fixed 224×224 tensor, are correspondingly stable — `forward` spans only 32.29–50.50 ms across all 30 requests.

**The optimisation this identified has been applied and measured.** See §3.3 for the change and §1a for what it did and did not achieve.

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

Gate 3 improves less because its Laplacian variance is deliberately excluded. Sharpness is a spatial derivative, so its value depends on resolution by definition, and it keeps its own existing 1024px path. That path is untouched, and a test asserts the variance does not move when the analysis resolution changes.

**Decision preservation.** 38 unit tests assert that the accept/reject verdict is identical to full resolution across images placed deliberately near each threshold. Because those images are synthetic and the claim concerns real clinical images, [`backend/scripts/verify_gate_downsampling.py`](../../backend/scripts/verify_gate_downsampling.py) runs the same comparison over a real APTOS directory:

```bash
python backend/scripts/verify_gate_downsampling.py aptos2019/train_images
```

**At the uncalibrated thresholds, this reported zero verdict changes across all 3,662 APTOS images** (790.7 s), with Laplacian deviation exactly 0.000000 and largest deviations of 0.0032 on aperture coverage and 0.30 on contrast standard deviation.

> [!IMPORTANT]
> **That result was measured at thresholds that rejected every genuine image**,
> so most comparisons never reached the contrast or illumination checks at all.
> Re-run after calibration, the check reported **one Gate 3 verdict change in
> 3,662 images (0.027%)**, on an image whose contrast sat on the cut-point.
>
> Two defects in the checker itself were found while reading that output, and
> both are fixed:
>
> - It **restated its thresholds as literals**, so it reported against
>   `contrast 18.0` after calibration had moved the value, and counted 2,502
>   images as "near" a boundary the system no longer used.
> - It took its deviations and margins from the **display-rounded** metrics, so
>   every margin it printed was quantised onto the rounding grid — which is why
>   it reported a closest margin of exactly `0.0000` for 2,502 separate images.
>
> The claim being checked also changed, deliberately. Zero flips is not a
> property any threshold through a continuous distribution can guarantee: some
> image is always arbitrarily close to the cut-point. The checker now verifies
> the stronger and falsifiable statement — **no image clear of its boundary
> changes verdict** — and reports the flip rate rather than suppressing it. A
> flip whose margin exceeds the deviation measured on that same image is not a
> boundary effect and still fails the run.
>
> **Outstanding:** a corpus re-run with the corrected checker, to restate this
> table against the calibrated thresholds. Until it lands, the figures above
> describe the superseded configuration and are labelled as such.

**One deviation needs explaining rather than glossing over.** The red/blue ratio shows a maximum absolute deviation of **127.85**, which looks alarming beside a threshold of 1.15. That metric is $r_{	ext{mean}} / (b_{	ext{mean}} + 10^{-6})$: on a very dark image with almost no blue signal it takes enormous values, so a small change in $b_{	ext{mean}}$ moves it a long way in absolute terms while leaving it orders of magnitude clear of the cutoff. A deviation that large can only arise where the ratio is already far above 1.15, which is why no verdict changed. The verifier now also reports deviation restricted to images *near* each threshold, plus the closest margin any image came to a boundary, so a future run demonstrates this directly instead of resting on the argument.

The end-to-end consequence is measured in §1 and §1a.

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
| Mean | 8.34 ms |
| Median | 8.15 ms |
| P95 | 10.12 ms |
| Min / Max | 7.46 / 10.69 ms |

Raw data in [`benchmark_timings.csv`](benchmark_timings.csv). The corresponding CPU forward pass is 33.82 ms — a 4× gap that is unsurprising and, as §2 shows, largely irrelevant to end-to-end response time.

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
