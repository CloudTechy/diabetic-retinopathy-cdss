# Model Training & Computational Environment Specification

## Metadata & Traceability
- **Research Project:** AI-Based Clinical Decision Support System for Early Detection of Diabetic Retinopathy
- **Author / Researcher:** Onyekelu Chukwuebuka Elochukwu (2024516020FN)
- **Related Research Objective:** Objective f (Train the CNN) & Objective e (Implement the CNN)
- **Git Commit:** `22cda2c` (Baseline)
- **Date Generated:** 2026-09-28
- **Evidence Files:** [`docs/chapter4/training_protocol.md`](file:///c:/Users/USER/Documents/TECH4MATION/diabetic-retinopathy-cdss/docs/chapter4/training_protocol.md), [`docs/chapter4/resource_benchmark.md`](file:///c:/Users/USER/Documents/TECH4MATION/diabetic-retinopathy-cdss/docs/chapter4/resource_benchmark.md)
- **Hardware/Software Environment:** Windows 11 Enterprise x64 / Intel(R) Core(TM) i7 / NVIDIA RTX Graphics / PyTorch 2.6

---

## 1. Hardware Infrastructure

| Hardware Component | Specification | Purpose in CDSS Development |
| :--- | :--- | :--- |
| **Central Processing Unit (CPU)** | Intel(R) Core(TM) i7-12700H @ 2.70 GHz (14 Cores, 20 Threads) | Data preprocessing, OpenCV validation pipeline, batch streaming, multi-threaded CPU inference. |
| **Graphics Processing Unit (GPU)** | NVIDIA GeForce RTX 3070 Ti Laptop GPU (8 GB GDDR6 VRAM, CUDA 12.4) | Accelerated model training, tensor backpropagation, batch matrix multiplications. |
| **System Memory (RAM)** | 32.0 GB DDR5 @ 4800 MHz | In-memory dataset caching and multi-worker DataLoader buffering. |
| **Storage Subsystem** | 1 TB NVMe PCIe Gen 4.0 SSD (Read: 5,000 MB/s, Write: 4,400 MB/s) | High-speed I/O for 40,000+ raw fundus photographs. |
| **Operating System** | Microsoft Windows 11 Enterprise (64-bit, Build 22631) | Primary development and clinical workstation platform. |

---

## 2. Software Frameworks, Runtimes & Libraries

| Category | Package / Tool | Version | Specific Functional Role |
| :--- | :--- | :---: | :--- |
| **Deep Learning Framework** | `torch` (PyTorch) | 2.6.0+ | Tensor operations, automatic differentiation, neural network execution. |
| **Vision Library** | `torchvision` | 0.21.0+ | EfficientNet-B0 pretrained architecture, ImageNet transforms. |
| **Scientific Computing** | `numpy` | 2.1.3 | Pixel matrix manipulation, array indexing, metric computation. |
| **Image Processing** | `pillow` (PIL) | 11.1.0 | Bitstream verification, format conversion, bicubic spatial resampling. |
| **Machine Learning Metrics** | `scikit-learn` | 1.6.1 | Confusion matrix computation, Quadratic Weighted Kappa ($\kappa$), F1. |
| **Visualization & Heatmaps** | `matplotlib` | 3.10.0 | Normalized confusion matrix visualization, colormap rendering. |
| **Backend Web Framework** | `fastapi` | 0.110.0 | RESTful API server routing inference requests. |
| **Database Engine** | `PostgreSQL` | 16.2 | Relational metadata storage and immutable audit logging. |
| **Frontend Framework** | `React` / `TypeScript` | 18.2 / 5.4 | Clinical browser user interface. |
| **Frontend Build Tool** | `Vite` | 5.4.21 | Hot module reloading and production minification. |

---

## 3. Determinism & Scientific Reproducibility Controls

To guarantee exact reproducibility across repeated training runs:
1. **Random Seeds:** `torch.manual_seed(42)`, `torch.cuda.manual_seed_all(42)`, `np.random.seed(42)`, `random.seed(42)`.
2. **Deterministic Algorithm Execution:** `torch.backends.cudnn.deterministic = True` and `torch.backends.cudnn.benchmark = False`.
3. **DataLoader Worker Seeding:** Worker initialization seeds set sequentially based on worker ID to eliminate non-deterministic multithreaded sample ordering.
