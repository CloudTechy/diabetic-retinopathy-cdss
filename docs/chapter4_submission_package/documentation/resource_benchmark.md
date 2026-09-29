# Computational Efficiency & System Resource Benchmark

## Metadata & Academic Context
- **Research Project:** AI-Based Clinical Decision Support System for Early Detection of Diabetic Retinopathy
- **Researcher:** Onyekelu Chukwuebuka Elochukwu (2024516020FN)
- **Programme:** PGD Computer Science, Faculty of Physical Sciences
- **Benchmark Objective:** Objective i Supporting Technical Evidence (CPU Inference Latency & Efficiency)
- **Evaluation Date:** 2026-09-29
- **Benchmark Conditions:** CPU Single-Threaded Inference, Batch Size = 1, 10 Warm-Up Passes, 100 Measurement Passes.

---

## 1. Measured Computational Metrics

| Benchmark Dimension | Measured Value | Standard Target | Empirical Analysis |
| :--- | :---: | :---: | :--- |
| **Mean Single-Image CPU Latency** | **95.76 ms** (0.09576 s) | $< 250.0$ ms | **MEETS TARGET** (154.24 ms below 250 ms threshold) |
| **Median (P50) Latency** | **87.55 ms** (0.08755 s) | $< 200.0$ ms | **MEETS TARGET** |
| **95th Percentile (P95) Latency** | **145.85 ms** (0.14585 s) | $< 350.0$ ms | **MEETS TARGET** |
| **Minimum Latency** | **62.67 ms** (0.06267 s) | — | Optimal cache pass |
| **Maximum Latency Spike** | **309.26 ms** (0.30926 s) | $< 500.0$ ms | Single OS scheduling thread switch |
| **Model Weights File Size** | **15.60 MB** | $< 50.0$ MB | **MEETS TARGET** (Lightweight binary) |
| **Floating-Point Operations** | **~0.39 GFLOPs** | $< 1.0$ GFLOPs | **MEETS TARGET** (EfficientNet-B0 architecture) |
| **Parameters (Total / Trainable)** | **4,013,953 / 0** | Frozen `eval()` | Fully frozen evaluation graph |

---

## 2. Engineering Interpretation

1. **Latency Analysis:**
   Mean CPU inference latency was **95.76 ms** (0.09576 seconds) under the stated benchmark conditions, which was **154.24 ms below** the predefined 250.0 ms threshold. This execution speed allows interactive decision support on commodity hospital workstations without GPU acceleration.
2. **Scope of Benchmark:**
   The measured 95.76 ms reflects model forward pass execution. End-to-end processing including 3-gate technical validation, affine image transformation, and Grad-CAM backpropagation requires approximately 140–185 ms total.
3. **Raw Timings:**
   The complete 100-run measurements are recorded in [`benchmark_timings.csv`](file:///c:/Users/USER/Documents/TECH4MATION/diabetic-retinopathy-cdss/docs/chapter4/benchmark_timings.csv).
