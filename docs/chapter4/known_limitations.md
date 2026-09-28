# Clinical & Technical Limitations Analysis

## Metadata & Traceability
- **Research Project:** AI-Based Clinical Decision Support System for Early Detection of Diabetic Retinopathy
- **Author / Researcher:** Onyekelu Chukwuebuka Elochukwu (2024516020FN)
- **Related Research Objective:** Objective h (Evaluation) & Objective i (Clinical governance)
- **Git Commit:** `22cda2c` (Baseline)
- **Date Approved:** 2026-09-28

---

## 1. Sub-Pixel Attenuation in Mild NPDR (Grade 1 vs Grade 0)

### Technical Analysis:
As demonstrated in the empirical held-out evaluation report ([`docs/chapter4/model_evaluation_report.md`](file:///c:/Users/USER/Documents/TECH4MATION/diabetic-retinopathy-cdss/docs/chapter4/model_evaluation_report.md)), the lowest per-class sensitivity was observed in **Grade 1 (Mild NPDR)** at **74.07%** (80/108 correctly classified), with 21 false negatives staged as Grade 0 (No DR).

### Pathological Rationale:
- **Physical Lesion Scale:** Early microaneurysms typically measure between $25\ \mu\text{m}$ and $50\ \mu\text{m}$ in diameter. On an uncompressed $2240 \times 1488$ pixel sensor, an isolated microaneurysm spans 3 to 6 pixels.
- **Resampling Distortion:** When downsampled to the standard $224 \times 224 \times 3$ input resolution required by EfficientNet-B0, solitary microaneurysms undergo spatial smoothing and bilinear interpolation attenuation, occasionally reducing their local gradient contrast below the activation floor of initial convolution layers.
- **CDSS Mitigation:** To protect patients, the system outputs the full continuous 5-class score distribution rather than a binary threshold. Even in the 21 misclassified cases, 18 retained a secondary Grade 1 score ($P \in [0.18, 0.35]$), prompting manual magnification during human clinician review.

---

## 2. 2D Monocular Photography vs. 3D Optical Coherence Tomography (OCT)

### Clinical Boundary:
- Diabetic Macular Edema (DME) is the leading cause of moderate visual acuity loss in diabetic patients.
- 2D color fundus photography can detect secondary surrogate signs of macular edema (such as hard exudate rings within one disc diameter of the fovea), but cannot directly measure cross-sectional retinal thickness, intraretinal fluid cysts, or subretinal fluid accumulation.
- **System Safeguard:** The CDSS report explicitly recommends that any patient presenting with hard exudates in the macular zone undergo confirmatory Spectral-Domain Optical Coherence Tomography (SD-OCT).

---

## 3. Field of View Constraints (45° Posterior Pole vs. Ultra-Widefield)

### Photographic Limitation:
- Standard desktop fundus photography captures a $45^\circ$ to $50^\circ$ monocular field centered on the fovea and optic disc.
- The International Clinical Diabetic Retinopathy (ICDR) 4-2-1 rule relies on assessing venous beading and hemorrhages across all 4 quadrants. While posterior pole arcades capture the majority of lesions, predominantly peripheral diabetic lesions (PPLs) occurring outside the $50^\circ$ field may be under-sampled compared to $200^\circ$ ultra-widefield imaging (e.g., Optos California).
- **System Safeguard:** The application provides laterality labeling (OD/OS) and camera model tracking to alert clinicians when peripheral coverage may require multi-field montage imaging.

---

## 4. Pupillary Mydriasis & Lens Opacity Dependency

### Technical Quality Boundary:
- In non-mydriatic screening without pharmacological dilation, small pupils ($< 3.0$ mm), patient fatigue, and nuclear cataracts cause significant peripheral vignetting and illumination attenuation.
- **Validation Gate 3 Safeguard:** The CDSS enforces strict automated gates for Laplacian blur variance ($\ge 60.0$) and illumination index ($0.20 \le \bar{Y} \le 0.85$). If cataract haze or pupil constriction violates these thresholds, automated inference is aborted, protecting clinicians from erroneous automated observations on ungradable images.

---

## 5. Camera Color Calibration & Inter-Device Variability

### Domain Shift Factor:
- Retinal datasets (EyePACS, APTOS, Messidor) aggregate imagery from diverse camera manufacturers (Topcon, Canon, Zeiss, Kowa), each with distinct sensor spectral response curves, illumination spectra, and internal sharpening algorithms.
- **Preprocessing Mitigation:** The pipeline utilizes robust color-space normalization, aspect ratio checking ($0.8 - 1.25$), and chromatic $R/B$ ratio filtering ($> 1.15$) to minimize camera domain shifts before tensor ingestion.
