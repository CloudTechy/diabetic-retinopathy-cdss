# Chapter Four Evidence & Reproduction Submission Package

> [!IMPORTANT]
> ## Verify this package before reading it
>
> **No installation required** — no virtual environment, no PyTorch, no
> repository checkout, no network. Python 3.8+ and the standard library:
>
> ```bash
> python VERIFY.py
> ```
>
> It recomputes the headline metrics from the raw predictions (Quadratic
> Weighted Kappa implemented from first principles, not imported), confirms the
> checkpoint digest, confirms the integration fixture is a genuine held-out
> APTOS image, confirms the split is leakage-free, and confirms every test count
> stated in these documents matches the committed log.
>
> [`VERIFICATION.md`](VERIFICATION.md) explains each check and carries a
> recorded transcript, so the package can be checked by eye without running
> anything.

**Programme:** Postgraduate Diploma (PGD) in Computer Science
**Faculty:** Faculty of Physical Sciences
**Project:** AI-Based Clinical Decision Support System for Early Detection of Diabetic Retinopathy
**Candidate:** Onyekelu Chukwuebuka Elochukwu (Reg No: 2024516020FN)
**Evidence run:** 2026-09-30 (clean rerun) — APTOS 2019, Google Colab Tesla T4, checkpoint `67d0b896…`

---

## 1. Summary

All model-performance artefacts derive from the documented leakage-free training and evaluation run, whose console transcript, per-epoch history, per-image predictions and checkpoint digest are all included and mutually consistent. System, interface and build evidence was produced separately and is identified by its corresponding provenance record in [`docs/chapter4/evidence_provenance.md`](docs/chapter4/evidence_provenance.md).

| Item | Value |
| :--- | :--- |
| **Trained weights** | EfficientNet-B0, 15.60 MB, SHA-256 `67d0b89641f08057126dd411e380b25575ef29f71ae37ee5796d472d9203dbf7` |
| **Held-out cohort** | $N = 525$ |
| **Quadratic Weighted Kappa** | **0.8658** |
| **Exact accuracy** | 84.00% (441 / 525) — see the note below |
| **Within-one-grade agreement** | 93.71% |
| **Referable DR** (grade ≥ 2) | Sensitivity **91.2%**, specificity **95.6%** |
| **Sight-threatening DR** (grade ≥ 3) | Sensitivity **68.2%**, NPV **95.4%** |
| **Any DR** (grade ≥ 1) | Sensitivity 97.6%, specificity 98.5% |
| **Argmax contradictions** | 0 / 525 |
| **Dataset** | 3,662 published APTOS 2019 records; **3,504 retained** after collapsing image-hash duplicate groups. Each row carries the SHA-256 of its real image bytes |
| **Test suite** | 266 passed, 1 skipped |
| **End-to-end CPU latency** | **180.41 ms** mean / 147.45 ms median / **342.49 ms** P95 — the canonical run (C) |

Every number in this table is checked against `docs/chapter4/clinical_metrics.json` and `docs/chapter4/cpu_end_to_end_benchmark.json` by a rule in the test suite, so it cannot drift from the artefacts silently.

> **On the headline metric.** Exact 5-class accuracy is the weakest available summary here, because the held-out cohort is 51.4% Grade 0 and the ICDR scale is ordinal. $\kappa$ and the referable-DR operating point are the meaningful figures. This is discussed in [`docs/chapter4/model_evaluation_report.md`](docs/chapter4/model_evaluation_report.md) §1.

### Disclosed limitations

1. **Partition contamination — resolved.** The original split leaked because APTOS's `duplicated_info.csv` is absent from the Kaggle download, so the grouping step silently no-opped. The split was rebuilt on image-hash groups and the model retrained. In the committed split **0 of 525 held-out images are byte-identical to a training image**, 0 of 526 validation images are, and **no hash group spans two partitions** (`clinical_metrics.json` → `leakage_audit`). Because nothing leaks, there is no clean-subset comparison to report and `leakage_adjusted` is `null`. See [`docs/chapter4/dataset_audit.md`](docs/chapter4/dataset_audit.md) §4.

2. **Latency is dominated by input handling, not inference.** End-to-end CPU latency is **180.41 ms mean / 342.49 ms P95** on the canonical run. Validation gates 2+3 cost **74.69 ms (41.4%)** — reduced from 59.1% by subsampling their statistics — while the model forward pass is **29.22 ms (16.2%)**. Latency scales with camera resolution, not with disease severity.

   Two earlier runs of the same harness are retained in [`docs/chapter4/resource_benchmark.md`](docs/chapter4/resource_benchmark.md) §1b purely as a **measurement of between-session variance**: two runs of the same harness and checkpoint over the same 30 images came out 1.41× apart in absolute time; their combined gates 2+3 share agreed to within 0.5 pp (41.9% and 41.4%), while individual stage shares differed by up to 2.4 pp (`gate3`). Cite the combined gate share and the ordering of the 5 largest stages (`gate2` > `encode` > `forward` > `compose` > `gate3`), not the milliseconds.

---

## 2. Layout

The archive is **repository-relative**: every file sits at the path it has in the project repository, so every link in every document resolves, and the test suite runs from `backend/` exactly as it does in the repository. Only the three files at the root are lifted out of `docs/chapter4/`.

```text
DR-CDSS_Chapter4_Evidence/
├── README.md                          # this file
├── VERIFY.py                          # six independent checks, stdlib only
├── VERIFICATION.md                    # what each check means + recorded transcript
├── PROGRESS_TRACKER.md
├── .env.example
├── .github/workflows/evidence-integrity-gate.yml   # the CI job that runs the gate
├── backend/
│   ├── main.py, requirements.txt, pytest.ini
│   ├── app/                           # the served application (gates, inference, reports, auth)
│   ├── models/weights/efficientnet_b0_dr.pth        # trained weights (15.60 MB, SHA-256 67d0b896…)
│   ├── scripts/                       # every script that produced or checks the evidence
│   └── tests/                         # the complete suite, incl. the integrity gate (rule groups A–Z)
│       └── fixtures/aptos_heldout_d1f1ea894da1.png  # a genuine held-out APTOS image
├── frontend/src/                      # the clinical UI sources
├── notebooks/colab_train_and_evaluate.py            # the pipeline that produced the run
└── docs/chapter4/
    ├── model_evaluation_report.md     # metrics, operating points, error structure
    ├── dataset_audit.md               # provenance, partition, duplicate audit
    ├── training_protocol.md           # hyperparameters + 15-epoch convergence ledger
    ├── training_environment.md        # hardware and versions, training and deployment
    ├── checkpoint_manifest.md         # topology, digest, runtime provenance enforcement
    ├── preprocessing_and_augmentation_spec.md
    ├── resource_benchmark.md          # CPU end-to-end benchmark + the T4 forward pass
    ├── known_limitations.md           # what this model cannot do
    ├── reproducibility_runbook.md     # three levels of reproduction
    ├── validation_module_spec.md      # the three technical-acceptance gates
    ├── database_schema.md             # all nine tables, transcribed from the model
    ├── architecture.md, api_contract.md, implementation_status.md
    ├── requirements_test_matrix.md, objective_traceability_matrix.md
    ├── system_test_report.md          # the test run, as recorded
    ├── screenshot_evidence_manifest.md
    ├── evidence_provenance.md         # every artefact and the script that produced it
    ├── DATASET_MANIFEST_README.md     # why no images ship; how to verify the manifest
    ├── verify_manifest_hashes.py      # verifies the manifest against YOUR APTOS copy
    ├── dataset_split_manifest.csv     # 3,504 records (2,453 train / 526 val / 525 test)
    ├── training_execution.log, epoch_history.csv, training_summary.json
    ├── held_out_predictions.csv       # 525 rows, full softmax distributions
    ├── evaluation_summary.json, clinical_metrics.json
    ├── cpu_end_to_end_benchmark.{json,csv}          # the canonical CPU run, 9 stages, 30 images
    ├── benchmark_timings.csv, benchmark_summary.json   # T4 forward pass only
    ├── gate_downsampling_verification.json          # 3,662 images, 1 boundary flip, 0 unexplained
    ├── validation_test_results.csv    # gate behaviour: 16 declared cases
    ├── dataset_split_audit.json, blur_threshold_calibration.json
    ├── test_execution.log             # the single authoritative suite run
    ├── test_environment_freeze.txt    # pip freeze of the environment that produced it
    ├── confusion_matrix.png, learning_curves.png
    ├── screenshots/                   # clinical UI captures against the running stack
    └── archive/                       # superseded documents, kept as correction records
```

---

## 3. Reading order

1. Run `python VERIFY.py`. Everything below assumes it passed.
2. [`docs/chapter4/model_evaluation_report.md`](docs/chapter4/model_evaluation_report.md) — the results and what they do and do not show.
3. [`docs/chapter4/dataset_audit.md`](docs/chapter4/dataset_audit.md) — how the split was contaminated, how it was rebuilt, and the audit of the manifest that ships.
4. [`docs/chapter4/known_limitations.md`](docs/chapter4/known_limitations.md) — read before citing any figure.
5. [`docs/chapter4/evidence_provenance.md`](docs/chapter4/evidence_provenance.md) — which script produced each file, and which scripts produced nothing.
6. [`docs/chapter4/archive/`](docs/chapter4/archive/) — earlier QA responses and runbooks, superseded and labelled as such. They record what was wrong; they are not current evidence.

The evidence integrity gate — `backend/tests/test_editor_integrity_gate.py` and `backend/tests/test_spec_doc_consistency.py` — encodes every finding from the QA rounds as an executable rule. It can be installed as a pre-commit hook (`python backend/scripts/integrity_gate.py --install-hook`) and the shipped workflow (`.github/workflows/evidence-integrity-gate.yml`) runs it on every push in CI; the archive carries that configuration, not a CI run record or proof that the hook is installed. The rules run unchanged from inside this archive.

---

## 4. Verification — run these yourself

### 4.1 Recompute every metric (no ML dependencies, < 5 seconds)

```bash
python backend/scripts/analyze_clinical_metrics.py
```

Reads `docs/chapter4/held_out_predictions.csv` and recomputes QWK (implemented from first principles), per-class sensitivity/specificity with Wilson confidence intervals, all three operating points, and the duplicate-leakage audit — using only the Python standard library.

### 4.2 Verify the checkpoint

```bash
sha256sum backend/models/weights/efficientnet_b0_dr.pth
# 67d0b89641f08057126dd411e380b25575ef29f71ae37ee5796d472d9203dbf7
```

```powershell
Get-FileHash backend\models\weights\efficientnet_b0_dr.pth -Algorithm SHA256
```

The running system enforces this same digest and refuses to serve on mismatch.

### 4.3 Verify the dataset manifest against real images

```bash
python docs/chapter4/verify_manifest_hashes.py <path>/aptos2019/train_images --sample 25
```

See [`docs/chapter4/DATASET_MANIFEST_README.md`](docs/chapter4/DATASET_MANIFEST_README.md) for why the images themselves are not redistributed.

### 4.4 Run the test suite from this archive

```bash
cd backend
python -m venv .venv
.venv/Scripts/python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe -m pytest tests/ -v
# Linux/macOS: .venv/bin/python in place of .venv/Scripts/python.exe
```

This is exactly how [`docs/chapter4/test_execution.log`](docs/chapter4/test_execution.log) was produced — from a fresh extraction of this archive, in a new virtual environment built from `backend/requirements.txt`. The log's header records the interpreter and `rootdir`; [`docs/chapter4/test_environment_freeze.txt`](docs/chapter4/test_environment_freeze.txt) is the `pip freeze` of that environment, and a rule in the suite checks it satisfies `requirements.txt`.

### 4.5 Build and run the application from this archive

The archive is also a runnable copy of the application. From the archive root:

```bash
cd frontend && npm ci && npm run build && npm run check:preflight && cd ..     # type-check + production bundle
docker compose config                               # validates the compose file
docker compose up --build                           # db + backend + frontend
# backend health: http://127.0.0.1:8000/api/v1/health   frontend: http://127.0.0.1:3000
```

On Windows, extract the archive to a short path (for example `C:/tmp`): paths longer than 260 characters break the extraction of nested `node_modules` binaries during `npm ci`. All compose variables have defaults, so no `.env` is required; `.env.example` carries the calibrated thresholds and a rule keeps it equal to `config.py`. [`docs/chapter4/build_verification.log`](docs/chapter4/build_verification.log) records these commands run from a clean extraction of this archive, ending with the backend's `/health` response — `inference_ready: true` and the SHA-256 the engine verified before loading — and the frontend answering HTTP 200. The `cap:*` scripts in `package.json` target a Capacitor Android project that is outside this evidence archive.

### 4.6 Cross-check the confusion matrix

`docs/chapter4/evaluation_summary.json` must agree with `docs/chapter4/confusion_matrix.png` and with §2 of `docs/chapter4/model_evaluation_report.md`:

```text
[[266,  2,  1,  0,  1],
 [  5, 33, 12,  0,  0],
 [  0, 11,105,  8, 15],
 [  0,  0,  5, 15,  6],
 [  1,  6,  9,  2, 22]]
```

Row sums: 270 / 50 / 139 / 26 / 40 = 525. Trace = 441 = 84.00%.

---

## 5. How this archive was built

`python backend/scripts/assemble_submission_package.py --build <zip>` copies every listed file byte-for-byte from the repository, extracts the result to a temporary directory, runs `VERIFY.py` there, and keeps the archive only if every check passes. Nothing in it is generated at build time. The archive's SHA-256 is content-deterministic: a rebuild of identical content is byte-identical.

---

## 6. Revision note

An earlier revision of this package reported $N = 544$, $\kappa = 0.9415$ and 86.40% accuracy, and shipped 15 placeholder PNGs as "sample test images". Those artefacts did not originate from a real training run: the manifest's recorded hashes matched **none** of the actual APTOS files, and it carried a `patient_id` column that APTOS 2019 does not publish. A later revision used a genuine run (2026-09-29) whose partition turned out to be contaminated by duplicate images. Both were replaced by the leakage-free 2026-09-30 run documented above, and the documents record each correction rather than concealing it.

---

## 7. Scope

This is a **research prototype supporting a PGD dissertation**. It is not a medical device, holds no regulatory clearance, has undergone no prospective clinical trial, and must not be used for patient care. All model output is decision *support* requiring professional review.
