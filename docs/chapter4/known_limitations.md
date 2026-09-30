# Clinical & Technical Limitations Analysis

## Metadata & Traceability
- **Research Project:** AI-Based Clinical Decision Support System for Early Detection of Diabetic Retinopathy
- **Author / Researcher:** Onyekelu Chukwuebuka Elochukwu (2024516020FN)
- **Related Research Objective:** Objective h (Evaluation) & Objective i (Clinical governance)
- **Date:** 2026-09-29
- **Basis:** Every figure cited below is recomputed from [`held_out_predictions.csv`](held_out_predictions.csv).

---

## 1. BLOCKING: the validation pipeline rejects every genuine image

> [!CAUTION]
> This is a defect in the shipped configuration, not a property of the data.

The first genuine run of the image-validation pipeline
([`validation_test_results.csv`](validation_test_results.csv)) rejected
**10 of 10 unmodified held-out APTOS images**. Nine failed Gate 3 with
`ERR_MOTION_OR_DEFOCUS_BLUR`; one failed Gate 1 on resolution.

| Image | Laplacian variance | Threshold | Verdict |
| :--- | ---: | ---: | :--- |
| `07a0e34c8d20` | 5.7 | 60.0 | REJECTED |
| `07d8db76b301` | 9.6 | 60.0 | REJECTED |
| `1623e8e3adc4` | 10.0 | 60.0 | REJECTED |
| `005b95c28852` | 11.0 | 60.0 | REJECTED |
| `0e0fc1d9810c` | 14.6 | 60.0 | REJECTED |
| `0f495d87656a` | 15.7 | 60.0 | REJECTED |
| `014508ccb9cb` | 16.3 | 60.0 | REJECTED |
| `0dc031c94225` | 18.2 | 60.0 | REJECTED |
| `0ceb222f6629` | 22.0 | 60.0 | REJECTED |

The highest-scoring genuine image reaches 22.0 against a threshold of 60.0. As
configured, the system would refuse to grade **every** real fundus photograph in
this corpus, and the classifier reported in `model_evaluation_report.md` would
never be reached in normal operation.

**Why this was not caught earlier.** Every previous check measured the gates
without asserting an expected outcome: the CPU benchmark timed them, and
`verify_gate_downsampling.py` compared each verdict against itself at two
resolutions — both consistent, both wrong. The defect only became visible when
`generate_validation_evidence.py` declared what each case *should* return.

**What it is not.** The gate discriminates correctly in the other direction: all
six derived negatives (blur, channel swap, flat field, letterbox, truncation,
non-image bytes) were rejected for the right reason, and the deliberately
blurred image scored 0.9 against 5.7–22.0 for unmodified ones. The ordering is
sound; the cut-point is in the wrong place.

**Required correction.** `LAPLACIAN_BLUR_THRESHOLD` must be calibrated from the
distribution of Laplacian variance over the real corpus rather than chosen a
priori, and the chosen percentile documented. That calibration needs the APTOS
images and has not been run. **Objective b cannot be claimed as met until it
is.**

## 2. Proliferative DR (Grade 4) is the weakest class — 55.0% sensitivity

### Empirical finding
Grade 4 has the lowest per-class sensitivity in the clean held-out cohort:
**55.0%** (22 / 40), 95% CI 39.8 – 69.3. Its 18 errors distribute as
9 → Grade 2, **6 → Grade 1**, 2 → Grade 3, **1 → Grade 0**.

Grade 3 is close behind at 57.7% (15 / 26, CI 38.9 – 74.5), and Grade 1 at
66.0% (33 / 50, CI 52.2 – 77.6).

In the superseded contaminated run Grade 2 was the weakest class at 53.3%; it
is now 75.5%. The ranking changed, so this section was rewritten rather than
renumbered.

### Why it matters more than the percentage suggests
The seven Grade 4 images placed at Grade 0 or Grade 1 are displaced by three or
four grade levels. Because quadratic weighted kappa penalises displacement by
the square of the distance, those seven cases account for most of the small
$\kappa$ decline relative to the contaminated run, even though accuracy,
macro-F1 and within-one-grade agreement all rose.

### Constraint
Grades 3 and 4 carry only 26 and 40 held-out cases respectively. The confidence
intervals are correspondingly wide — roughly 30 percentage points — so these
point estimates indicate a weakness without measuring its size precisely. A
larger severe-grade cohort would be required to narrow them, and APTOS does not
contain one.

## 3. Sub-pixel attenuation in Mild NPDR (Grade 1)

### Empirical finding
Grade 1 sensitivity is **72.7%** (40 / 55). Only **3** Grade 1 cases were misclassified as Grade 0 — the clinically worst direction for this class. In those 3 cases the Grade 1 score was retained at 0.092, 0.206 and 0.298 respectively, in each case as the second-ranked class.

The larger share of Grade 1 error (11 cases) is over-calling to Grade 2, which is the safe direction.

### Pathological rationale
- Early microaneurysms measure roughly 25–50 µm. On a typical multi-megapixel fundus sensor an isolated microaneurysm spans only a few pixels.
- Resizing to $224 \times 224$ applies spatial smoothing that can attenuate a solitary lesion's local gradient contrast below what the early convolutional layers respond to.

**Caveat on this explanation.** The resolution argument is a plausible mechanism consistent with the error pattern, not a demonstrated cause. Establishing it would require an ablation at higher input resolution, which was not run.

---

## 4. Wide confidence intervals on the severe grades

Grades 3 and 4 carry only **29** and **45** held-out cases respectively. The resulting Wilson intervals are correspondingly wide:

| Grade | Sensitivity | 95% CI | Interval width |
| :---: | :---: | :---: | :---: |
| 3 (Severe NPDR) | 58.6% | 40.7 – 74.5 | 33.8 pts |
| 4 (Proliferative DR) | 62.2% | 47.6 – 74.9 | 27.3 pts |

These point estimates should be treated as indicative. A cohort several times larger in the severe grades would be needed to state them with useful precision.

---

## 5. No external validation

All reported performance comes from a held-out split of a **single cohort** (APTOS 2019, Aravind Eye Hospital, Tamil Nadu). The model has never been evaluated on a different population, camera fleet, or grading panel.

Retinal datasets differ systematically in sensor spectral response, illumination, compression and grading convention, and cross-dataset performance drops are well documented in this field. **No claim about generalisation is made or supported by this work.** This is the single largest constraint on clinical interpretation of these results.

---

## 6. Duplicate leakage in the original split — resolved by retraining

The partition used for all superseded results was contaminated: 3,662 records
over 3,534 unique image hashes, 48 duplicate groups spanning partitions (27 test
images byte-identical to a training image, 17 validation, 6 test-to-validation),
and 30 duplicate groups carrying conflicting severity labels.

**This has been corrected.** The split was rebuilt by hashing image bytes,
excluding the 30 label-conflicting groups, keeping one representative per
remaining group, and asserting zero hash overlap between partitions before
training. EfficientNet-B0 was then retrained from ImageNet initialisation on the
result. The current held-out cohort has **0 / 525** images byte-identical to any
training image.

What remains a limitation is the cost of the correction: **30 distinct images
(0.85%) were excluded** because their duplicate groups disagreed about the
severity label and the disagreement cannot be adjudicated from the image data.
Nine of those groups disagreed by two or more grade levels; one was labelled
both Mild NPDR and Proliferative DR. Their identifiers and conflicting grades
are listed in [`dataset_split_audit.json`](dataset_split_audit.json) so a
clinician could adjudicate them and restore them to a future split.

## 7. Input handling, not inference, dominates response time

End-to-end CPU latency is **212.54 ms mean / 179.68 ms median / 373.39 ms P95**
over 30 held-out images on a 4-thread x86_64 CPU
([`cpu_end_to_end_benchmark.json`](cpu_end_to_end_benchmark.json)). All figures
in this section come from that single post-optimisation run.

The structure of the number is the limitation worth stating:

| Stage | Mean (ms) | Share |
| :--- | ---: | ---: |
| `gate2` (aperture & relevance) | 71.11 | 33.5% |
| `encode` (PNG serialisation of the overlay) | 39.83 | 18.7% |
| `forward` (the model itself) | 33.82 | **15.9%** |
| `gate3` (quality) | 23.41 | 11.0% |
| `compose` (Grad-CAM overlay) | 22.86 | 10.8% |
| `preprocess` | 13.12 | 6.2% |
| `gate1`, `gradcam`, `read` | 7.17 | 3.3% |

The model forward pass is **15.9%** of the request. Validation gates 2 and 3
together cost **94.52 ms (44.5%)** — already reduced by computing their
distribution statistics on a nearest-neighbour subsample, a change verified
decision-preserving across all 3,662 APTOS images. Serialising the Grad-CAM PNG
costs more than inference.

Two consequences:

1. **Latency scales with input resolution, not with disease severity.** `gate2`
   runs a **42.05 ms median against a 182.37 ms P95** — a 4.3× range across
   APTOS's varied image dimensions — because the reduction step must still read
   every source pixel. The model stages, operating on a fixed $224 	imes 224$
   tensor, are stable by comparison: `forward` spans 28.64–42.77 ms across all
   30 requests. A site with higher-resolution cameras will see proportionally
   slower responses with no change in classification behaviour.

2. **Further reducible cost is identified but not removed.** Gates 1, 2 and 3
   each convert the full-resolution image independently. Decoding once and
   sharing a single reduced copy is the next available gain. It is not
   implemented, and no benefit from it is claimed.

Excluding image I/O, the compute-only mean is 110.32 ms. The measurement was
taken on a shared cloud CPU; a dedicated clinical workstation would likely be
faster, but none was benchmarked, so no figure for one is offered.

## 8. 2D fundus photography cannot assess macular oedema

- Diabetic Macular Edema is a leading cause of moderate visual acuity loss in diabetic patients, and the ICDR severity grade does not encode it.
- Colour fundus photography shows surrogate signs (hard exudate rings near the fovea) but cannot measure retinal thickness, intraretinal fluid or subretinal fluid.
- **Scope:** DME assessment is outside this system's scope. The system reports five-class ICDR severity only, and makes no management or imaging recommendation.

---

## 9. Field-of-view constraints

- Standard fundus photography captures a 45°–50° field centred on the fovea and optic disc.
- The ICDR 4-2-1 rule assesses haemorrhages and venous beading across four quadrants. Predominantly peripheral lesions outside the captured field may be under-sampled relative to ultra-widefield imaging.
- **Safeguard:** the application records laterality (OD/OS) so a clinician can identify where multi-field montage imaging is warranted.

---

## 10. Image quality dependency

- In non-mydriatic screening, small pupils, patient fatigue and lens opacity produce vignetting and illumination loss.
- **Gate 3 safeguard:** the pipeline enforces Laplacian blur variance $\ge 60.0$ and bounded illumination. Images violating these thresholds are rejected before inference rather than graded unreliably — the system declines rather than guesses.

---

## 11. Scope of the artefact

This is a **research prototype supporting a PGD dissertation**. It is not a medical device, carries no regulatory clearance (FDA, CE, MHRA or otherwise), has undergone no prospective clinical trial, and must not be used for patient care. Every output is positioned as decision *support* requiring clinician review and sign-off, and the system records that review as part of the audit trail.
