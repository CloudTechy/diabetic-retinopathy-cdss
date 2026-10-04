# Clinical & Technical Limitations Analysis

## Metadata & Traceability
- **Research Project:** AI-Based Clinical Decision Support System for Early Detection of Diabetic Retinopathy
- **Author / Researcher:** Onyekelu Chukwuebuka Elochukwu (2024516020FN)
- **Related Research Objective:** Objective h (Classification-performance evaluation)
- **Last Revised:** 2026-10-04
- **Basis:** Every figure cited below is recomputed from [`held_out_predictions.csv`](held_out_predictions.csv).

---

## 1. RESOLVED: three admission thresholds were never calibrated

> [!NOTE]
> **Closed 2026-10-01.** All 10 unmodified held-out images are now ACCEPTED and
> all 6 derived negatives REJECTED, each for the reason its derivation predicts
> ([`validation_test_results.csv`](validation_test_results.csv)). The record
> below is kept because the defect was real and shipped, and because the reason
> it went unnoticed matters more than the fix.

The first genuine run of the image-validation pipeline rejected
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

**It was not one threshold but three.** Correcting the blur cut-point left five
of ten genuine images still rejected, all on `ERR_LOW_CONTRAST`. Measured over
the 2,979 development images:

| Setting | A priori | Calibrated (1st percentile) | Genuine images the a priori value rejected |
| :--- | ---: | ---: | ---: |
| `LAPLACIAN_BLUR_THRESHOLD` | 60.0 | **4.3** | 2,963 / 2,979 — **99.5%** |
| `CONTRAST_THRESHOLD` | 18.0 | **8.8** | 1,890 / 2,979 — **63.4%** |
| `MIN_IMAGE_DIMENSION` | 512 | **480** | 38 / 2,979 — 1.3% |

Each is the 1st percentile of its own distribution, measured over the
**training and validation partitions only** — choosing an operating point on the
held-out test set would leak it, and
[`calibrate_blur_threshold.py`](../../backend/scripts/calibrate_blur_threshold.py)
refuses a calibration that includes it. The percentile is the declared
judgement and is recorded with the distribution it came from in
[`validation_module_spec.md`](validation_module_spec.md).

`MIN_IMAGE_DIMENSION` deserves its own note: at 512 it refused images the model
was **trained on** after resizing to 224×224. A gate that rejects its own
training distribution is incoherent regardless of how few images it touches.

**What remains a limitation.** The thresholds admit 99% of *this* corpus by
construction. They are not a clinical quality standard, and a site with
different cameras would need to recalibrate. The gates establish that an image
is gradeable by this system, not that it is diagnostically adequate.

**Objective b is met.** All 16 declared cases behave as declared.

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
Grade 1 sensitivity is **66.0%** (33 / 50). **5** Grade 1 cases were misclassified as Grade 0 — the clinically worst direction for this class. In 3 of those 5 cases the Grade 1 score was retained at 0.092, 0.206 and 0.298 respectively, in each case as the second-ranked class.

The larger share of Grade 1 error (11 cases) is over-calling to Grade 2, which is the safe direction.

### Pathological rationale
- Early microaneurysms measure roughly 25–50 µm. On a typical multi-megapixel fundus sensor an isolated microaneurysm spans only a few pixels.
- Resizing to $224 \times 224$ applies spatial smoothing that can attenuate a solitary lesion's local gradient contrast below what the early convolutional layers respond to.

**Caveat on this explanation.** The resolution argument is a plausible mechanism consistent with the error pattern, not a demonstrated cause. Establishing it would require an ablation at higher input resolution, which was not run.

---

## 4. Wide confidence intervals on the severe grades

Grades 3 and 4 carry only **26** and **40** held-out cases respectively. The resulting Wilson intervals are correspondingly wide:

| Grade | Sensitivity | 95% CI | Interval width |
| :---: | :---: | :---: | :---: |
| 3 (Severe NPDR) | 57.7% | 38.9 – 74.5 | 35.6 pts |
| 4 (Proliferative DR) | 55.0% | 39.8 – 69.3 | 29.5 pts |

These point estimates should be treated as indicative. A cohort several times larger in the severe grades would be needed to state them with useful precision.

---

## 5. No external validation

All reported performance comes from a held-out split of a **single cohort** (APTOS 2019, Aravind Eye Hospital, Tamil Nadu). The model has never been evaluated on a different population, camera fleet, or grading panel.

Retinal datasets differ systematically in sensor spectral response, illumination, compression and grading convention, and cross-dataset performance drops are well documented in this field. **No claim about generalisation is made or supported by this work.** This is the single largest constraint on clinical interpretation of these results.

---

## 6. Duplicate leakage in the original split — resolved by retraining

The partition used for all superseded results was contaminated: 3,662 records
over 3,534 unique image hashes, 48 duplicate groups spanning partitions (27 test
images byte-identical to a training image, 17 validation, 6 test-to-validation —
image counts, which exceed 48 because 1 group span all three partitions),
and 30 duplicate groups carrying conflicting severity labels. Those figures
recompute from that manifest, archived as
[`archive/dataset_split_manifest.superseded_65edf3d.csv`](archive/dataset_split_manifest.superseded_65edf3d.csv).

**This has been corrected.** The split was rebuilt by hashing image bytes,
excluding the 30 label-conflicting groups, keeping one representative per
remaining group, and asserting zero hash overlap between partitions before
training. EfficientNet-B0 was then retrained from ImageNet initialisation on the
result. The current held-out cohort has **0 / 525** images byte-identical to any
training image.

What remains a limitation is the cost of the correction: **30 duplicate groups — 62 images (1.7% of the 3,662 published records) — were excluded** because their duplicate groups disagreed about the
severity label and the disagreement cannot be adjudicated from the image data.
Nine of those groups disagreed by two or more grade levels; one was labelled
both Mild NPDR and Proliferative DR. Their identifiers and conflicting grades
are listed in [`dataset_split_audit.json`](dataset_split_audit.json) so a
clinician could adjudicate them and restore them to a future split.

## 7. Input handling, not inference, dominates response time

End-to-end CPU latency is **180.41 ms mean / 147.45 ms median / 342.49 ms P95**
over 30 held-out images on a 4-thread x86_64 CPU
([`cpu_end_to_end_benchmark.json`](cpu_end_to_end_benchmark.json)). All figures
in this section come from that single run. The harness times gates 2 and 3 and
continues regardless of their verdict, so every image executes the full path
whatever the thresholds; it measures the cost of the accepted path, not the
rejection path.

**The absolute figure is a property of the machine as much as of the system.**
Two runs of the same harness and checkpoint over the same 30 images, differing in
which machine Colab allocated, came out **1.41× apart** (254.31 ms and 180.41 ms);
their combined gates 2+3 share agreed to within 0.5 pp (41.9% and 41.4%), while individual stage shares differed by up to 2.4 pp (`gate3`)
([`resource_benchmark.md`](resource_benchmark.md) §1b). The combined gate share and
the ordering of the 5 largest stages (`gate2` > `encode` > `forward` > `compose` > `gate3`) are the durable finding; the milliseconds are an order of
magnitude.

**The tail is the part worth stating.** NFR-01 sets a 350 ms budget and is
specified on the mean, which passes in all three runs. The **P95 breaches it in
two of three** (373.39, 447.95, 342.49 ms). Whether one request in twenty
exceeds the budget depends on the hardware, and the requirement as written
cannot see that, because it examines only the average.

The structure of the number is the limitation worth stating:

| Stage | Mean (ms) | Share |
| :--- | ---: | ---: |
| `gate2` (aperture & relevance) | 59.03 | 32.7% |
| `encode` (PNG serialisation of the overlay) | 30.61 | 17.0% |
| `forward` (the model itself) | 29.22 | **16.2%** |
| `compose` (Grad-CAM overlay) | 18.20 | 10.1% |
| `gate3` (quality) | 15.66 | 8.7% |
| `read` | 12.97 | 7.2% |
| `preprocess` | 10.36 | 5.7% |
| `gate1`, `gradcam` | 2.94 | 1.7% |

The model forward pass is **16.2%** of the request. Validation gates 2 and 3
together cost **74.69 ms (41.4%)** — already reduced by computing their
distribution statistics on a nearest-neighbour subsample. Serialising the
Grad-CAM PNG costs more than inference.

The decision-preservation evidence for that subsampling has been re-measured
with the corrected checker ([`resource_benchmark.md`](resource_benchmark.md)
§3.3): **one** Gate 3 verdict change across 3,662 images, on an image whose
contrast margin (0.0129) was smaller than the deviation measured on that same
image (0.0162) — inside the measurement band. **No image clear of its boundary
changed verdict**, and zero flips were unexplained.

Two consequences:

1. **Latency scales with input resolution, not with disease severity.** `gate2`
   runs a **32.43 ms median against a 161.89 ms P95** — a 5.0× range across
   APTOS's varied image dimensions — because the reduction step must still read
   every source pixel. The model stages, operating on a fixed $224 \times 224$
   tensor, are stable by comparison: `forward` spans 25.86–35.33 ms across all
   30 requests. A site with higher-resolution cameras will see proportionally
   slower responses with no change in classification behaviour. This is also
   where the P95 sits relative to the budget: the tail is resolution, not the
   model.

2. **Further reducible cost is identified but not removed.** Gates 1, 2 and 3
   each convert the full-resolution image independently. Decoding once and
   sharing a single reduced copy is the next available gain. It is not
   implemented, and no benefit from it is claimed.

Excluding image I/O, the compute-only mean is 89.02 ms (`compute_only_mean_ms` in `cpu_end_to_end_benchmark.json`). The measurement was
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
- **Gate 3 safeguard:** the pipeline enforces Laplacian blur variance $\ge 4.3$ and bounded illumination. Images violating these thresholds are rejected before inference rather than graded unreliably — the system declines rather than guesses.

---

## 11. Scope of the artefact

This is a **research prototype supporting a PGD dissertation**. It is not a medical device, carries no regulatory clearance (FDA, CE, MHRA or otherwise), has undergone no prospective clinical trial, and must not be used for patient care. Every output is positioned as decision *support* requiring clinician review and sign-off, and the system records that review as part of the audit trail.
