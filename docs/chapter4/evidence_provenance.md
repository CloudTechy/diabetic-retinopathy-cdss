# Evidence Provenance — which script produced which file

A reviewer asking *"what produced this number?"* should get one answer, not a
shortlist. This file gives that answer for every artefact in the package.

It exists because the package ships several scripts that *look* authoritative
but produced none of the committed evidence. That ambiguity is the same problem
that made the rejected package hard to audit, and naming the producer is
cheaper than arguing about it.

---

## The run that produced the evidence

**`scripts/colab_train_and_evaluate.py`**, executed on Google Colab with a
Tesla T4, 2026-09-29. One invocation produced the manifest, the trained
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
| `dataset_sample_and_manifest/dataset_split_manifest.csv` | `colab_train_and_evaluate.py` |
| `checkpoint/efficientnet_b0_dr.pth` | `colab_train_and_evaluate.py` |
| `logs_and_metrics/training_execution.log` | `colab_train_and_evaluate.py` |
| `logs_and_metrics/epoch_history.csv` | `colab_train_and_evaluate.py` |
| `logs_and_metrics/training_summary.json` | `colab_train_and_evaluate.py` |
| `logs_and_metrics/held_out_predictions.csv` | `colab_train_and_evaluate.py` |
| `logs_and_metrics/evaluation_summary.json` | `colab_train_and_evaluate.py` |
| `logs_and_metrics/benchmark_timings.csv` | `colab_train_and_evaluate.py` (T4, forward pass) |
| `logs_and_metrics/benchmark_summary.json` | `colab_train_and_evaluate.py` (T4, forward pass) |
| `visualizations/learning_curves.png` | `colab_train_and_evaluate.py` |
| `visualizations/confusion_matrix.png` | `colab_train_and_evaluate.py` |

## Derived afterwards, from those artefacts

| Artefact | Produced by | Reads |
| :--- | :--- | :--- |
| `logs_and_metrics/clinical_metrics.json` | `scripts/analyze_clinical_metrics.py` | `held_out_predictions.csv`, `dataset_split_manifest.csv` |
| `logs_and_metrics/cpu_end_to_end_benchmark.json` | `scripts/benchmark_cpu_end_to_end.py` | the checkpoint + held-out images |
| `logs_and_metrics/cpu_end_to_end_benchmark.csv` | `scripts/benchmark_cpu_end_to_end.py` | same run, flat form |
| `logs_and_metrics/gate_downsampling_verification.json` | `scripts/verify_gate_downsampling.py` | all 3,662 APTOS images |
| `logs_and_metrics/validation_test_results.csv` | `scripts/generate_validation_evidence.py` | held-out images + stated derivations |
| `screenshots/01`–`08` | `frontend/scripts/capture_screenshots.js` | the built frontend |
| `screenshots/09_confusion_matrix_empirical.png` | copy of `visualizations/confusion_matrix.png` | — |

---

## Scripts that produced **none** of the committed evidence

These are standalone equivalents of stages the Colab notebook performs
internally. They are included because they let the pipeline be re-run outside
Colab, and because the reviewer asked to see the training and evaluation code.
**They did not generate any file in this package.**

| Script | Standalone equivalent of | Distinguishable because |
| :--- | :--- | :--- |
| `scripts/train_efficientnet_b0.py` | the training stage | emits `Epoch \| Train Loss \| Val Loss \| Val Acc`, not the header above |
| `scripts/evaluate_model.py` | the evaluation stage | writes its own field ordering |

Both training implementations are genuine — each defines a `torch.utils.data.Dataset`
whose `__getitem__` opens a real image file, and iterates a real `DataLoader`.
The distinction here is **provenance**, not authenticity.

---

## Verification tooling (produces no evidence)

| Script | Purpose |
| :--- | :--- |
| `scripts/integrity_gate.py` | Runs the integrity gate locally; `--install-hook` enforces it per commit |
| `scripts/test_editor_integrity_gate.py` | 63 assertions encoding the QA review's requirements |
| `scripts/test_spec_doc_consistency.py` | Fails if a document quotes a threshold the code does not enforce |
| `scripts/benchmark_resources.py` | Forward-pass-only benchmark; selects CUDA when present |
| `dataset_sample_and_manifest/verify_manifest_hashes.py` | Checks manifest hashes against your own APTOS copy |

---

## Removed, and why

| File | Reason |
| :--- | :--- |
| `validation_test_results.csv` *(previous version)* | Fabricated. Zero of its ten rows cited an actual held-out image: two were training images presented as held-out, six used identifiers absent from APTOS entirely (`NONRET-XRAY-01`, `CORRUPT-BYTE-01`, …), and its metrics were demonstration constants copied from the frontend mock. Regenerate with `generate_validation_evidence.py`. |
| `sample_test_images/` | 640×480 synthetic placeholders, eight byte-identical, matching no real APTOS file. See `dataset_sample_and_manifest/README.md`. |
| `hash_verification_output.txt` | Reported those placeholders as "VERIFIED MATCH". |
| `create_evaluated_checkpoint.py` | Built a randomly-initialised network and saved it to the production weights path. |
| `generate_dataset_manifest.py` | Invented the N = 8,000 multi-cohort manifest. |
| `prepare_submission_package.py` | Synthesised benchmark timings and wrote the 86.40% / 0.9415 reports. |
| `generate_aptos_manifest.py` | Superseded by `build_clean_split.py`. It keyed de-duplication on `duplicated_info.csv`, which the Kaggle competition download does not contain, so the step silently grouped nothing. |
