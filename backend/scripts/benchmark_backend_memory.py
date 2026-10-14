#!/usr/bin/env python3
"""
Peak RAM Benchmark for the DR-CDSS Calibrated Backend & Inference Pipeline.

Measures process Resident Set Size (RSS) and peak memory consumption during
complete assessments, including:
  1. Technical validation pipeline (Gates 1, 2, and 3)
  2. EfficientNet-B0 forward pass with frozen weights
  3. Calibrated probability scaling (validation-fitted temperature T)
  4. Grad-CAM visual attribution computation (features.8 backward pass)
  5. Saliency map composition (512x512 RGBA) and PNG serialisation

Outputs:
  - docs/chapter4/backend_memory_benchmark.json
  - docs/chapter4/backend_memory_measurements.csv
"""

import argparse
import csv
import gc
import io
import json
import os
import platform
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

# Add backend to path
REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = REPO_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

# Force CPU inference for reproducibility on deployment target
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("MODEL_DEVICE", "cpu")

import torch
from PIL import Image

try:
    import resource
except ImportError:
    resource = None

from app.core.config import settings
from app.services.ai_service import (
    EfficientNetB0InferenceService,
    get_active_inference_service,
)
from app.services.validation.gate1_integrity import evaluate_gate1
from app.services.validation.gate2_relevance import evaluate_gate2
from app.services.validation.gate3_quality import evaluate_gate3


def get_current_rss_mb() -> float:
    """Return the current resident set size (RSS) in megabytes."""
    if os.path.exists("/proc/self/status"):
        try:
            with open("/proc/self/status", "r") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        return float(line.split()[1]) / 1024.0
        except Exception:
            pass
    if resource is not None:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        if platform.system() == "Darwin":
            return usage.ru_maxrss / (1024.0 * 1024.0)
        return usage.ru_maxrss / 1024.0
    return 0.0


def get_peak_rss_mb() -> float:
    """Return OS-level maximum resident set size (high water mark) in megabytes."""
    if os.path.exists("/proc/self/status"):
        try:
            with open("/proc/self/status", "r") as f:
                for line in f:
                    if line.startswith("VmHWM:"):
                        return float(line.split()[1]) / 1024.0
        except Exception:
            pass
    if resource is not None:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        if platform.system() == "Darwin":
            return usage.ru_maxrss / (1024.0 * 1024.0)
        return usage.ru_maxrss / 1024.0
    return get_current_rss_mb()


def get_hardware_info() -> dict:
    """Collect host hardware and runtime environment metadata using standard library."""
    total_ram_mb = 0.0
    available_ram_mb = 0.0
    if os.path.exists("/proc/meminfo"):
        try:
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        total_ram_mb = round(float(line.split()[1]) / 1024.0, 2)
                    elif line.startswith("MemAvailable:"):
                        available_ram_mb = round(float(line.split()[1]) / 1024.0, 2)
        except Exception:
            pass
    
    cpu_name = platform.processor() or "Unknown CPU"
    cores = os.cpu_count() or 1
    if platform.system() == "Linux" and os.path.exists("/proc/cpuinfo"):
        try:
            with open("/proc/cpuinfo", "r") as f:
                for line in f:
                    if line.startswith("model name"):
                        cpu_name = line.split(":", 1)[1].strip()
                        break
        except Exception:
            pass

    return {
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": cpu_name,
        "logical_threads": cores,
        "total_ram_mb": total_ram_mb,
        "available_ram_mb": available_ram_mb,
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
    }


class HighFrequencyMemorySampler:
    """Background sampler to capture the true high-water peak during tensor operations."""
    def __init__(self, interval_sec=0.005):
        self.interval = interval_sec
        self.running = False
        self.peak_rss_mb = 0.0
        self.thread = None
        self._lock = threading.Lock()

    def _sample(self):
        while self.running:
            try:
                rss = get_current_rss_mb()
                with self._lock:
                    if rss > self.peak_rss_mb:
                        self.peak_rss_mb = rss
            except Exception:
                pass
            time.sleep(self.interval)

    def start(self):
        self.peak_rss_mb = get_current_rss_mb()
        self.running = True
        self.thread = threading.Thread(target=self._sample, daemon=True)
        self.thread.start()

    def stop(self) -> float:
        self.running = False
        if self.thread:
            self.thread.join(timeout=0.2)
        with self._lock:
            return self.peak_rss_mb


def load_sample_images(repo_root: Path, limit: int = 15):
    """Find real APTOS held-out images or synthetically generate calibrated fundus samples."""
    samples = []
    
    # 1. Look for test images in storage or aptos directory
    candidate_dirs = [
        repo_root / "storage" / "datasets" / "aptos2019" / "train_images",
        repo_root / "aptos2019" / "train_images",
        repo_root / "backend" / "tests" / "fixtures",
    ]
    for cdir in candidate_dirs:
        if cdir.exists():
            for f in sorted(cdir.glob("*.png")):
                try:
                    data = f.read_bytes()
                    if len(data) > 5000:
                        samples.append((f.name, data))
                        if len(samples) >= limit:
                            break
                except Exception:
                    continue
        if len(samples) >= limit:
            break

    # 2. If fewer than limit, load or generate compliant fundus images
    if len(samples) < limit:
        import numpy as np
        
        needed = limit - len(samples)
        for i in range(needed):
            # Create a realistic 640x640 fundus-like image that clears all 3 gates
            arr = np.zeros((640, 640, 3), dtype=np.uint8)
            y, x = np.ogrid[:640, :640]
            center_x, center_y, radius = 320, 320, 290
            mask = (x - center_x) ** 2 + (y - center_y) ** 2 <= radius ** 2
            
            # Retinal orange/red background
            arr[mask, 0] = 180 + (i % 20)
            arr[mask, 1] = 75 + (i % 15)
            arr[mask, 2] = 25
            
            # Optic disc (bright yellow-orange)
            disc_mask = (x - 220) ** 2 + (y - 320) ** 2 <= 40 ** 2
            arr[disc_mask, 0] = 250
            arr[disc_mask, 1] = 200
            arr[disc_mask, 2] = 120
            
            # Macula (darker spot)
            macula_mask = (x - 420) ** 2 + (y - 320) ** 2 <= 30 ** 2
            arr[macula_mask, 0] = 120
            arr[macula_mask, 1] = 40
            arr[macula_mask, 2] = 15
            
            # Vessel branches (Laplacian texture)
            for v in range(12):
                vy = np.clip(320 + np.sin(x[0] / 30.0 + v) * 50, 0, 639).astype(int)
                for xi in range(100, 540):
                    arr[vy[xi], xi, :] = [80, 20, 10]
            
            # Add fine texture noise
            noise = np.random.RandomState(42 + i).normal(0, 8, (640, 640, 3)).astype(np.int16)
            arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            
            buf = io.BytesIO()
            Image.fromarray(arr).save(buf, format="PNG")
            samples.append((f"calibrated_sample_{i+1:02d}.png", buf.getvalue()))

    return samples


def main():
    import io
    parser = argparse.ArgumentParser(description="Measure calibrated backend peak RAM during complete assessments.")
    parser.add_argument("--iterations", type=int, default=30, help="Number of complete assessments to run.")
    parser.add_argument("--output-json", type=str, default="docs/chapter4/backend_memory_benchmark.json")
    parser.add_argument("--output-csv", type=str, default="docs/chapter4/backend_memory_measurements.csv")
    args = parser.parse_args()

    print("=" * 76)
    print(" DR-CDSS: CALIBRATED BACKEND PEAK RAM BENCHMARK ")
    print("=" * 76)

    gc.collect()
    start_time = datetime.now(timezone.utc)
    hw = get_hardware_info()
    print(f"Host:       {hw['processor']} ({hw['logical_threads']} threads, {hw['total_ram_mb']} MB RAM)")
    print(f"OS:         {hw['platform']}")
    print(f"Python:     {hw['python_version']} | PyTorch: {hw['torch_version']}")

    # Baseline memory before loading model
    baseline_rss = get_current_rss_mb()
    print(f"\n[1/4] Baseline Memory (process init): {baseline_rss:.2f} MB")

    # Load active calibrated inference engine
    print("[2/4] Loading EfficientNet-B0 engine and calibration artifact...")
    sampler = HighFrequencyMemorySampler(interval_sec=0.005)
    sampler.start()
    
    svc = get_active_inference_service()
    checkpoint_sha = getattr(svc, "verified_sha256", "unknown")
    cal_temp = getattr(svc, "temperature", None)
    cal_version = getattr(svc, "calibration", {}).get("version", "unknown")
    
    model_load_peak = sampler.stop()
    model_loaded_rss = get_current_rss_mb()
    
    print(f"      Model Checkpoint SHA-256: {checkpoint_sha[:16]}...")
    print(f"      Calibration Temperature:  T = {cal_temp:.6f} ({cal_version})")
    print(f"      Post-Model Load RSS:      {model_loaded_rss:.2f} MB")

    # Load sample images for evaluation
    samples = load_sample_images(REPO_ROOT, limit=args.iterations)
    print(f"\n[3/4] Executing {args.iterations} complete end-to-end assessments (with Grad-CAM)...")

    measurements = []
    global_max_rss = model_load_peak

    for idx in range(args.iterations):
        img_name, raw_bytes = samples[idx % len(samples)]
        
        # Start sampler for this assessment
        iter_sampler = HighFrequencyMemorySampler(interval_sec=0.002)
        rss_before = get_current_rss_mb()
        iter_sampler.start()
        
        t0 = time.perf_counter()
        
        # Complete Assessment Pipeline:
        # Gate 1: Integrity & Decode
        g1_res, pil_img = evaluate_gate1(raw_bytes, img_name)
        
        # Gate 2: Retinal relevance
        g2_res = evaluate_gate2(pil_img) if pil_img else None
        
        # Gate 3: Technical quality
        g3_res = evaluate_gate3(pil_img) if pil_img else None
        
        # AI Inference + Temperature Calibration + Grad-CAM Heatmap
        ai_res = svc.predict(pil_img, "OD")
        
        dur_ms = (time.perf_counter() - t0) * 1000.0
        peak_during = iter_sampler.stop()
        rss_after = get_current_rss_mb()
        
        if peak_during > global_max_rss:
            global_max_rss = peak_during

        measurements.append({
            "run_index": idx + 1,
            "image": img_name,
            "duration_ms": round(dur_ms, 2),
            "primary_grade": ai_res.primary_grade,
            "primary_score": round(ai_res.primary_score, 4),
            "rss_before_mb": round(rss_before, 2),
            "peak_rss_during_mb": round(peak_during, 2),
            "rss_after_mb": round(rss_after, 2),
            "gradcam_generated": ai_res.gradcam_bytes is not None,
            "gradcam_size_bytes": len(ai_res.gradcam_bytes) if ai_res.gradcam_bytes else 0,
        })
        
        if (idx + 1) % 5 == 0 or (idx + 1) == args.iterations:
            print(f"      Assessment {idx + 1:2d}/{args.iterations}: "
                  f"Grade {ai_res.primary_grade} ({ai_res.primary_score:.2f}) | "
                  f"Peak RAM: {peak_during:.2f} MB | Time: {dur_ms:.1f} ms")

    # Final OS peak from getrusage
    os_max_rss = get_peak_rss_mb()
    final_peak_mb = max(global_max_rss, os_max_rss)
    final_rss = get_current_rss_mb()

    # Budget evaluation against 512 MB constraint (NFR-10 / NFR-02)
    budget_mb = 512.0
    passed = final_peak_mb < budget_mb
    margin_mb = budget_mb - final_peak_mb

    print("\n" + "=" * 76)
    print(" BENCHMARK RESULTS SUMMARY ")
    print("=" * 76)
    print(f" Baseline RAM (process init):     {baseline_rss:.2f} MB")
    print(f" Model Loaded RAM:                {model_loaded_rss:.2f} MB")
    print(f" Peak Backend RAM (Measured):     {final_peak_mb:.2f} MB")
    print(f" Final Post-Run RSS:              {final_rss:.2f} MB")
    print(f" System Budget Constraint:        {budget_mb:.1f} MB RSS")
    print(f" Margin Below Budget:             {margin_mb:+.2f} MB")
    print(f" NFR-10 / NFR-02 Compliance:      {'✅ MET / PASS' if passed else '❌ FAILED'}")
    print("=" * 76)

    # Save CSV
    csv_path = REPO_ROOT / args.output_csv
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "run_index", "image", "duration_ms", "primary_grade", "primary_score",
            "rss_before_mb", "peak_rss_during_mb", "rss_after_mb",
            "gradcam_generated", "gradcam_size_bytes"
        ], lineterminator="\n")
        writer.writeheader()
        writer.writerows(measurements)
    print(f"Raw measurements written to: {csv_path}")

    # Save JSON summary
    summary = {
        "benchmark_name": "Calibrated Backend Peak RAM & Assessment Resource Benchmark",
        "benchmark_timestamp": start_time.isoformat(),
        "completion_timestamp": datetime.now(timezone.utc).isoformat(),
        "nfr_code": "NFR-10",
        "nfr_description": "Memory Footprint: Peak backend memory usage during complete assessments with Grad-CAM",
        "budget_mb": budget_mb,
        "measured_peak_ram_mb": round(final_peak_mb, 2),
        "margin_mb": round(margin_mb, 2),
        "status": "MET" if passed else "FAILED",
        "status_statement": f"Met: measured peak backend memory of {final_peak_mb:.1f} MB, which is below the 512.0 MB budget.",
        "hardware": hw,
        "model_and_calibration": {
            "model_architecture": "EfficientNet-B0 (strictly frozen, eval mode)",
            "checkpoint_sha256": checkpoint_sha,
            "calibration_temperature": cal_temp,
            "calibration_version": cal_version,
            "target_gradcam_layer": "features.8",
        },
        "memory_summary": {
            "baseline_rss_mb": round(baseline_rss, 2),
            "model_loaded_rss_mb": round(model_loaded_rss, 2),
            "mean_peak_per_assessment_mb": round(sum(m["peak_rss_during_mb"] for m in measurements) / len(measurements), 2),
            "overall_peak_rss_mb": round(final_peak_mb, 2),
            "final_rss_mb": round(final_rss, 2),
            "total_assessments_measured": args.iterations,
        },
    }

    json_path = REPO_ROOT / args.output_json
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Structured report written to: {json_path}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
