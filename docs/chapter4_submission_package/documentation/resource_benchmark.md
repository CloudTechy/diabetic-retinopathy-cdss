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

Two distinct pieces of work are outstanding, and neither is done:

**(a) CPU forward-pass latency.** [`backend/scripts/benchmark_resources.py`](../../backend/scripts/benchmark_resources.py) already selects the device automatically (`cuda` if available, else `cpu`) and takes no arguments. Running it unchanged on a CPU-only machine, or in the backend container, therefore produces the CPU figure directly:

```bash
python backend/scripts/benchmark_resources.py
```

**(b) End-to-end request latency.** This is **not implemented**. The script times the forward pass alone; it has no mode that exercises decode → three-gate validation → preprocessing → forward → Grad-CAM → heatmap composition. Producing the number a clinician actually experiences requires extending the script to time that full path, or instrumenting the `/api/v1/assessments` handler directly.

Once (a) and (b) exist, the CPU end-to-end figure should become the headline in §1, with the T4 forward-pass figure retained below it as a labelled training-environment reference point. Reporting both, clearly distinguished, is the honest presentation; reporting the T4 number as though it described the clinical workstation would not be.
