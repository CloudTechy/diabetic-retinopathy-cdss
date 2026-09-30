# Clinical & Technical Limitations Analysis

## Metadata & Traceability
- **Research Project:** AI-Based Clinical Decision Support System for Early Detection of Diabetic Retinopathy
- **Author / Researcher:** Onyekelu Chukwuebuka Elochukwu (2024516020FN)
- **Related Research Objective:** Objective h (Evaluation) & Objective i (Clinical governance)
- **Date:** 2026-09-29
- **Basis:** Every figure cited below is recomputed from [`held_out_predictions.csv`](held_out_predictions.csv).

---

## 1. Moderate NPDR (Grade 2) is the weakest class — 53.3% sensitivity

### Empirical finding
Grade 2 has the lowest per-class sensitivity in the held-out cohort: **53.3%** (80 / 150), 95% CI 45.4–61.1. Its 70 errors distribute as:

| Predicted as | Count |
| :--- | :---: |
| Grade 4 (Proliferative DR) | 27 |
| Grade 1 (Mild NPDR) | 25 |
| Grade 3 (Severe NPDR) | 15 |
| Grade 0 (No DR) | 3 |

### Analysis
Grade 2 is the ICDR scale's widest and least sharply bounded category — "more than microaneurysms but less than severe". Its boundary with Grade 1 below and Grade 3 above is a matter of lesion count and distribution, which is precisely the information most degraded by downsampling to $224 \times 224$.

Two observations soften the finding:

- **122 of 150 Grade 2 cases (81.3%) were assigned some grade $\ge 2$.** The misclassification is one of severity placement within the scale, not a failure to register abnormality. This is a statement about classifier behaviour, not about referral.
- In **23 of the 27** Grade-2-called-Grade-4 cases, Grade 2 remained the model's second-ranked class, so the correct grade was present in the output distribution.

### Mitigation in the CDSS
The system surfaces the full 5-class score distribution rather than a single label, so a reviewing clinician sees the competing hypothesis rather than an unqualified assertion.

---

## 2. Sub-pixel attenuation in Mild NPDR (Grade 1)

### Empirical finding
Grade 1 sensitivity is **72.7%** (40 / 55). Only **3** Grade 1 cases were misclassified as Grade 0 — the clinically worst direction for this class. In those 3 cases the Grade 1 score was retained at 0.092, 0.206 and 0.298 respectively, in each case as the second-ranked class.

The larger share of Grade 1 error (11 cases) is over-calling to Grade 2, which is the safe direction.

### Pathological rationale
- Early microaneurysms measure roughly 25–50 µm. On a typical multi-megapixel fundus sensor an isolated microaneurysm spans only a few pixels.
- Resizing to $224 \times 224$ applies spatial smoothing that can attenuate a solitary lesion's local gradient contrast below what the early convolutional layers respond to.

**Caveat on this explanation.** The resolution argument is a plausible mechanism consistent with the error pattern, not a demonstrated cause. Establishing it would require an ablation at higher input resolution, which was not run.

---

## 3. Wide confidence intervals on the severe grades

Grades 3 and 4 carry only **29** and **45** held-out cases respectively. The resulting Wilson intervals are correspondingly wide:

| Grade | Sensitivity | 95% CI | Interval width |
| :---: | :---: | :---: | :---: |
| 3 (Severe NPDR) | 58.6% | 40.7 – 74.5 | 33.8 pts |
| 4 (Proliferative DR) | 62.2% | 47.6 – 74.9 | 27.3 pts |

These point estimates should be treated as indicative. A cohort several times larger in the severe grades would be needed to state them with useful precision.

---

## 4. No external validation

All reported performance comes from a held-out split of a **single cohort** (APTOS 2019, Aravind Eye Hospital, Tamil Nadu). The model has never been evaluated on a different population, camera fleet, or grading panel.

Retinal datasets differ systematically in sensor spectral response, illumination, compression and grading convention, and cross-dataset performance drops are well documented in this field. **No claim about generalisation is made or supported by this work.** This is the single largest constraint on clinical interpretation of these results.

---

## 5. Byte-level duplicate leakage in the split (disclosed, measured, immaterial)

27 of 549 held-out images (4.92%) are byte-identical to a training image, because the de-duplication step depended on `duplicated_info.csv`, which is absent from the Kaggle competition download. See [`dataset_audit.md`](dataset_audit.md) §4.

Measured effect: accuracy 77.78% on the affected images vs 78.74% on the clean 522, and clean-subset $\kappa$ = 0.877818 vs full-cohort 0.877747. The condition did not inflate any reported metric.

**Future work:** regenerate the split with grouping keyed on the SHA-256 already present in the manifest, and re-train.

---

## 6. Input handling, not inference, dominates response time

End-to-end CPU latency is **164.79 ms mean / 137.21 ms median / 277.31 ms P95** over 30 held-out images on a 4-thread x86_64 CPU ([`resource_benchmark.md`](resource_benchmark.md) §1). All figures in this section are from that same post-optimisation run.

The structure of the number is the limitation worth stating. The model forward pass is **29.24 ms (17.7%)**. Validation gates 2 and 3 together cost **72.42 ms (43.9%)** — reduced from 59.1% by computing their distribution statistics on a nearest-neighbour subsample, a change verified decision-preserving across all 3,662 APTOS images. PNG serialisation of the Grad-CAM overlay is now **30.18 ms (18.3%)**, costing more than the forward pass.

Two consequences:

1. **Latency scales with input resolution, not with disease severity.** `gate2` runs a **32.33 ms median against a 142.99 ms P95** — a 4.4× range across APTOS's varied image dimensions — because the reduction step must still read every source pixel. The model stages, operating on a fixed $224 	imes 224$ tensor, are stable by comparison: `forward` spans only 26.49–36.49 ms across all 30 requests. A site with higher-resolution cameras will see proportionally slower responses with no change in classification behaviour.

2. **Further reducible cost is identified but not removed.** Gates 1, 2 and 3 each convert the full-resolution image independently. Decoding once and sharing a single reduced copy is the next available gain. It is not implemented, and no benefit from it is claimed.

The measurement was taken on a shared cloud CPU. A dedicated clinical workstation would likely be faster, but none was benchmarked, so no figure for one is offered.

---

## 7. 2D fundus photography cannot assess macular oedema

- Diabetic Macular Edema is a leading cause of moderate visual acuity loss in diabetic patients, and the ICDR severity grade does not encode it.
- Colour fundus photography shows surrogate signs (hard exudate rings near the fovea) but cannot measure retinal thickness, intraretinal fluid or subretinal fluid.
- **Scope:** DME assessment is outside this system's scope. The system reports five-class ICDR severity only, and makes no management or imaging recommendation.

---

## 8. Field-of-view constraints

- Standard fundus photography captures a 45°–50° field centred on the fovea and optic disc.
- The ICDR 4-2-1 rule assesses haemorrhages and venous beading across four quadrants. Predominantly peripheral lesions outside the captured field may be under-sampled relative to ultra-widefield imaging.
- **Safeguard:** the application records laterality (OD/OS) so a clinician can identify where multi-field montage imaging is warranted.

---

## 9. Image quality dependency

- In non-mydriatic screening, small pupils, patient fatigue and lens opacity produce vignetting and illumination loss.
- **Gate 3 safeguard:** the pipeline enforces Laplacian blur variance $\ge 60.0$ and bounded illumination. Images violating these thresholds are rejected before inference rather than graded unreliably — the system declines rather than guesses.

---

## 10. Scope of the artefact

This is a **research prototype supporting a PGD dissertation**. It is not a medical device, carries no regulatory clearance (FDA, CE, MHRA or otherwise), has undergone no prospective clinical trial, and must not be used for patient care. Every output is positioned as decision *support* requiring clinician review and sign-off, and the system records that review as part of the audit trail.
