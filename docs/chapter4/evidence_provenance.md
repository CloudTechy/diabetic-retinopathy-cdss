# Evidence Provenance — which script produced which file

A reviewer asking *"what produced this number?"* should get one answer, not a
shortlist. This file gives that answer for every artefact in the package.

It exists because the package ships several scripts that *look* authoritative
but produced none of the committed evidence. That ambiguity is the same problem
that made the rejected package hard to audit, and naming the producer is
cheaper than arguing about it.

---

## The run that produced the evidence

**`notebooks/colab_train_and_evaluate.py`**, executed on Google Colab with a
Tesla T4, 2026-09-30 (the leakage-free rerun). One invocation produced the manifest, the trained
checkpoint, the training transcript, the held-out predictions, the T4 benchmark
and both plots.

It is identifiable from the artefacts themselves: `training_execution.log`
opens with the column header

```text
 Ep |   TrLoss |    VLoss |   VAcc |   VQWK |    VF1 |         LR | Status
```

which only this script emits.

| Artefact | Produced by |
| :--- | :--- |
| `docs/chapter4/dataset_split_manifest.csv` | `colab_train_and_evaluate.py` |
| `backend/models/weights/efficientnet_b0_dr.pth` | `colab_train_and_evaluate.py` |
| `docs/chapter4/training_execution.log` | `colab_train_and_evaluate.py` |
| `docs/chapter4/epoch_history.csv` | `colab_train_and_evaluate.py` |
| `docs/chapter4/training_summary.json` | `colab_train_and_evaluate.py` |
| `docs/chapter4/held_out_predictions.csv` | `colab_train_and_evaluate.py` |
| `docs/chapter4/evaluation_summary.json` | `colab_train_and_evaluate.py` |
| `docs/chapter4/benchmark_timings.csv` | `colab_train_and_evaluate.py` (T4, forward pass) |
| `docs/chapter4/benchmark_summary.json` | `colab_train_and_evaluate.py` (T4, forward pass) |
| `docs/chapter4/learning_curves.png` | `colab_train_and_evaluate.py` |
| `docs/chapter4/confusion_matrix.png` | `colab_train_and_evaluate.py` |

## Derived afterwards, from those artefacts

| Artefact | Produced by | Reads |
| :--- | :--- | :--- |
| `docs/chapter4/clinical_metrics.json` | `backend/scripts/analyze_clinical_metrics.py` | `held_out_predictions.csv`, `dataset_split_manifest.csv` |
| `docs/chapter4/cpu_end_to_end_benchmark.json` | `backend/scripts/benchmark_cpu_end_to_end.py` | checkpoint `67d0b89641f0…` over 30 held-out images of the clean split (mean 1,798 KB), the same 30 as run B; calibrated thresholds in `config.py` |
| `docs/chapter4/cpu_end_to_end_benchmark.csv` | `backend/scripts/benchmark_cpu_end_to_end.py` | same run, flat form |
| `docs/chapter4/benchmark_history/cpu_end_to_end_benchmark_2026-09-29_baseline_pre_optimisation.json` | `backend/scripts/benchmark_cpu_end_to_end.py`, as committed at git `d05ef60b4033` (2026-09-29) | checkpoint `8ee14d7591a8…` — the **withdrawn** contaminated-run checkpoint, pinned in `config.py` at that commit — over 30 held-out images of the superseded split (mean 1,807 KB) |
| `docs/chapter4/benchmark_history/cpu_end_to_end_benchmark_2026-09-30_post_optimisation.json` | `backend/scripts/benchmark_cpu_end_to_end.py`, as committed at git `bba9ce43948e` (2026-09-30) | checkpoint `8ee14d7591a8…` (withdrawn; pinned at that commit) over the same 30 superseded-split images (mean 1,807 KB) |
| `docs/chapter4/benchmark_history/cpu_end_to_end_benchmark_2026-09-30_run_A_a_priori_thresholds.json` | `backend/scripts/benchmark_cpu_end_to_end.py`, as committed at git `422dcb72c96f` (2026-09-30) | committed in the same commit that installed the clean checkpoint `67d0b89641f0…`, but its image set is that of the earlier runs (mean 1,807 KB, the superseded split) and its `pytorch` field records a CUDA build on `device: cpu`; which checkpoint that session loaded cannot be established from the artefacts |
| `docs/chapter4/benchmark_history/cpu_end_to_end_benchmark_2026-10-01_run_B_calibrated.json` | `backend/scripts/benchmark_cpu_end_to_end.py`, as committed at git `1a7b5d251e6b` (2026-10-01) | checkpoint `67d0b89641f0…` over 30 held-out images of the clean split (mean 1,798 KB); `config.py` at that commit still held the a-priori threshold values, which the harness does not act on |
| `docs/chapter4/gate_downsampling_verification.json` | `backend/scripts/verify_gate_downsampling.py` over all 3,662 APTOS images (not re-run since: the images are not in this package). Its `thresholds_in_force` lists `RETINAL_MAX_COVERAGE` because that setting was declared when it ran; no gate ever read it, so the recorded verdicts are unaffected. The red-share floor in force then was the same 0.36 that is now `RETINAL_RED_SHARE_MIN`. | all 3,662 APTOS images |
| `docs/chapter4/validation_test_results.csv` | `backend/scripts/generate_validation_evidence.py` | held-out images + stated derivations. VAL-14's derivation label reads "2.2:1 panorama": the operation reduces the height to 45% of the original, so the measured `gate2_aspect_ratio` (2.938 for that source) is the authoritative value and the label describes the intent |
| `docs/chapter4/blur_threshold_calibration.json` | `backend/scripts/calibrate_blur_threshold.py` | the 2,979 train+val images named in `dataset_split_manifest.csv` |
| `docs/chapter4/dataset_split_audit.json` | `backend/scripts/build_clean_split.py` | the APTOS image bytes + `train.csv` labels |
| `docs/chapter4/archive/dataset_split_manifest.superseded_65edf3d.csv` | `notebooks/colab_train_and_evaluate.py` as committed at git `65edf3d6d6aa` (2026-09-29): the contaminated split, kept so the leakage figures in `known_limitations.md` §6 can be recomputed | the APTOS image bytes |
| `VERIFY.py` | hand-written; produces no evidence | reads the artefacts above and recomputes their headline figures. Stdlib only, so a reviewer needs no environment. Its recorded transcript is `VERIFICATION.md`. |
| `docs/chapter4/test_execution.log` | `python -m pytest tests/ -v` from `backend/` of a **fresh extraction of this archive**, in a new virtual environment built from `backend/requirements.txt`; its header records the interpreter and `rootdir` | the committed suite; the single authoritative run |
| `docs/chapter4/test_environment_freeze.txt` | `pip freeze` in that same virtual environment | — |
| `docs/chapter4/build_verification.log` | `npm ci && npm run build`, `npm run check:preflight` (the browser pre-check executed on synthetic inputs by `frontend/scripts/check_preflight.cjs`), `docker compose config` and `docker compose up --build` run from a **clean extraction of this archive**, followed by the backend `/health` response (`inference_ready` and the verified checkpoint SHA-256) and the frontend's HTTP status; recorded verbatim | the shipped build manifests |
| `docs/chapter4/container_environment_freeze.txt` | `pip freeze` and `python --version` run inside the backend container started by that compose check | the backend image built from `backend/Dockerfile` |
| `docs/chapter4/screenshots/01`–`08` | `frontend/scripts/capture_live_screenshots.js`, driving the running stack (real authentication, calibrated gates, digest-verified checkpoint). The rejection frame is written as `04b_validation_stepper_rejected.png` by `frontend/scripts/capture_rejection.js` alone, which fires the shutter only on the tick the rejected state is observed (an earlier revision of `capture_live_screenshots.js` also attempted it and threw on the server-side rejection path, which left its non-2xx guard unexecuted; that step is removed); the committed file is renamed `04b_fail_closed_rejection_worklist.png` to describe the frame it holds | the running application |
| `docs/chapter4/screenshots/09_confusion_matrix_empirical.png` | copy of `docs/chapter4/confusion_matrix.png` | — |

---

## Scripts that produced **none** of the committed evidence

These are standalone equivalents of stages the Colab notebook performs
internally. They are included because they let the pipeline be re-run outside
Colab, and because the reviewer asked to see the training and evaluation code.
**They did not generate any file in this package.**

| Script | Standalone equivalent of | Distinguishable because |
| :--- | :--- | :--- |
| `backend/scripts/train_efficientnet_b0.py` | the training stage | emits `Epoch \| Train Loss \| Val Loss \| Val Acc`, not the header above |
| `backend/scripts/evaluate_model.py` | the evaluation stage | writes its own field ordering |

Both training implementations are genuine — each defines a `torch.utils.data.Dataset`
whose `__getitem__` opens a real image file, and iterates a real `DataLoader`.
The distinction here is **provenance**, not authenticity.

---

## Verification tooling (produces no evidence)

| Script | Purpose |
| :--- | :--- |
| `backend/scripts/integrity_gate.py` | Runs the integrity gate locally; `--install-hook` enforces it per commit |
| `backend/tests/test_editor_integrity_gate.py` | The evidence integrity gate: one executable rule per QA finding, rule groups A–Z and beyond |
| `backend/tests/test_spec_doc_consistency.py` | Fails if a document quotes a threshold the code does not enforce |
| `backend/scripts/benchmark_resources.py` | Forward-pass-only benchmark; selects CUDA when present |
| `docs/chapter4/verify_manifest_hashes.py` | Checks manifest hashes against your own APTOS copy |

---

## Removed, and why

| File | Reason |
| :--- | :--- |
| `validation_test_results.csv` *(previous version)* | Fabricated. Zero of its ten rows cited an actual held-out image: two were training images presented as held-out, six used identifiers absent from APTOS entirely (`NONRET-XRAY-01`, `CORRUPT-BYTE-01`, …), and its metrics were demonstration constants copied from the frontend mock. Regenerate with `generate_validation_evidence.py`. |
| `sample_test_images/` | 640×480 synthetic placeholders, eight byte-identical, matching no real APTOS file. See `DATASET_MANIFEST_README.md`. |
| `hash_verification_output.txt` | Reported those placeholders as "VERIFIED MATCH". |
| `create_evaluated_checkpoint.py` | Built a randomly-initialised network and saved it to the production weights path. |
| `generate_dataset_manifest.py` | Invented the N = 8,000 multi-cohort manifest. |
| `prepare_submission_package.py` | Synthesised benchmark timings and wrote the 86.40% / 0.9415 reports. |
| `generate_aptos_manifest.py` | Superseded by `build_clean_split.py`. It keyed de-duplication on `duplicated_info.csv`, which the Kaggle competition download does not contain, so the step silently grouped nothing. |
