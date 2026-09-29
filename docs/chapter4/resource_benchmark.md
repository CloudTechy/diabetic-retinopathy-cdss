# Computational Efficiency & System Resource Benchmark

## Metadata & Academic Context
- **Research Project:** AI-Based Clinical Decision Support System for Early Detection of Diabetic Retinopathy
- **Researcher:** Onyekelu Chukwuebuka Elochukwu (2024516020FN)
- **Programme:** PGD Computer Science, Faculty of Physical Sciences
- **Benchmark Objective:** Objective i supporting evidence (inference latency & efficiency)
- **Execution Date:** 2026-09-29
- **Raw Evidence:** [`benchmark_timings.csv`](benchmark_timings.csv) (100 rows), [`benchmark_summary.json`](benchmark_summary.json)

---

## ⚠ Scope of this measurement — read first

This benchmark was captured **on the Google Colab Tesla T4 GPU used for training**, and it times **the model forward pass only**.

It therefore does **not** characterise the deployment target. The CDSS serves inference on **CPU** (`MODEL_DEVICE=cpu`), and an end-to-end clinical request additionally performs JPEG/PNG decode, the three-stage technical validation pipeline, tensor preprocessing, Grad-CAM backpropagation, and heatmap composition — none of which are included below.

**A CPU end-to-end benchmark on the deployment configuration has not yet been run.** Until it is, no claim is made in this thesis about clinical workstation response time. This gap is recorded in [`known_limitations.md`](known_limitations.md) and the procedure for closing it is given in §4.

---

## 1. Measured Metrics (Tesla T4, forward pass only)

Conditions: `torch.inference_mode()`, batch size 1, 10 warm-up passes, 100 measured passes, timed with CUDA events and explicit `torch.cuda.synchronize()`.

| Benchmark Dimension | Measured Value |
| :--- | :---: |
| **Mean forward-pass latency** | **8.36 ms** |
| Median (P50) | 8.34 ms |
| 95th percentile (P95) | 8.86 ms |
| Minimum | 7.91 ms |
| Maximum | 9.11 ms |
| Standard deviation (approx.) | 0.23 ms |
| **Model weights file size** | **15.60 MB** (16,358,249 bytes) |
| **Total parameters** | 4,013,953 |
| **Trainable parameters at inference** | **0** (frozen, `eval()`, `requires_grad=False`) |
| Theoretical FLOPs (EfficientNet-B0 @ 224²) | ~0.39 GFLOPs |

The distribution is tight — the full range spans 1.2 ms across 100 runs — which is expected for a fixed-shape forward pass on a dedicated accelerator with no host-side work in the timed region.

---

## 2. Held-Out Evaluation Throughput

A second, independent timing is available from the evaluation stage, which is closer to a realistic workload because it includes image loading and preprocessing:

| Measure | Value |
| :--- | :---: |
| Images processed | 549 |
| Total wall-clock time | 76.0 s |
| **Mean per-image, including PIL decode + preprocessing** | **138.4 ms** |

This still ran on the T4, so it remains a GPU figure — but the ~130 ms gap between it and the 8.36 ms forward pass shows that on this workload, image handling dominates model execution by more than an order of magnitude. That is the practical reason a CPU forward-pass number alone would not have predicted end-to-end latency either.

---

## 3. Efficiency Argument (device-independent)

The claims that survive regardless of device:

1. **The checkpoint is small.** 15.60 MB is trivially deployable, fits in container layers without special handling, and loads in well under a second.
2. **The compute cost is low.** ~0.39 GFLOPs per image is at the bottom of the modern CNN range; EfficientNet-B0 was selected over deeper backbones precisely for this.
3. **The inference graph is fully frozen.** Zero trainable parameters at serve time means no runtime adaptation, no drift between requests, and a prediction attributable to a fixed, hash-verified artefact.

---

## 4. Closing the CPU Gap

A dedicated harness now exists: [`backend/scripts/benchmark_cpu_end_to_end.py`](../../backend/scripts/benchmark_cpu_end_to_end.py).

It pins the device to CPU (it clears `CUDA_VISIBLE_DEVICES` before importing torch, so it cannot accidentally reproduce the GPU-measurement error this section exists to correct) and times the full request path **stage by stage**:

```text
read -> gate1 -> gate2 -> gate3 -> preprocess -> forward -> gradcam -> compose -> encode
```

Per-stage reporting is the point. A single total tells you a request is slow but not which part to fix.

### Running it

Requires the APTOS images. In a Colab **CPU** runtime:

```bash
pip install -q pydantic-settings
cd diabetic-retinopathy-cdss
python backend/scripts/benchmark_cpu_end_to_end.py --images-dir aptos2019/train_images --runs 30
```

It writes `cpu_end_to_end_benchmark.json` and `.csv` into `docs/chapter4/`. Once those exist, their total should replace §1 as the headline, with the T4 forward-pass figure retained below as a labelled training-environment reference point.

#### Fallback: synthetic mode

Where the 9.51 GB dataset is not available, `--synthetic` draws fundus-like images instead:

```bash
python backend/scripts/benchmark_cpu_end_to_end.py --synthetic --runs 30
```

What that does and does not license:

| Stage | On synthetic input |
| :--- | :--- |
| `preprocess`, `forward`, `gradcam`, `compose`, `encode` | **Valid.** These depend only on tensor shape and model topology, both identical to a real request. |
| `read`, `gate1` (decode), `gate2`, `gate3` | **Approximate.** These depend on file size and pixel statistics. |

The generated images are calibrated rather than arbitrary: without added noise a synthetic fundus encodes to ~0.03 MB, roughly ninety times smaller than a real file, which would make decode timings meaningless. The default noise sigma of 0.6 yields **2.72 MB** at 2048×1536 against the **2.66 MB** APTOS average — within 2%. All three validation gates pass on the generated images, so the full request path executes.

The script prints a `compute-only subtotal` and records `compute_only_mean_ms` in its JSON. **If reporting synthetic results, cite that figure, not the total**, and say that it is synthetic. Real images remain preferable.

### Already fixed as a result of building this harness

Profiling the path exposed that Grad-CAM heatmap composition built its 512x512 RGBA overlay with a nested Python loop — **262,144 interpreter iterations per request**, measured at **650-700 ms on CPU**. That single stage cost roughly an order of magnitude more than the model forward pass it accompanies.

It is now vectorised (`viridis_rgba_array`), taking **52 ms** — about **12x faster**, removing ~600 ms from every clinical request. Output is byte-identical, verified by 13 tests in `backend/tests/test_gradcam_colormap.py` against the original per-pixel implementation.

### Still outstanding

The measured CPU end-to-end total. The harness exists and the dominant bottleneck is fixed, but the benchmark has **not yet been executed on real fundus images**, so no total is recorded here. **Until it is, no clinical-workstation latency claim is made anywhere in this thesis.**
