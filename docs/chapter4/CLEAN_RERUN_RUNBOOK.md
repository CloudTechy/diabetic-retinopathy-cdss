# Clean Rerun Runbook — the one remaining step

Everything in the senior QA review has been actioned **except** the item that
needs a GPU: retraining on the leakage-free split and regenerating the
empirical artefacts.

This is a single Colab session. Nothing here requires a decision — the split
logic, the assertions and the artefact layout are already fixed in code.

---

## Why a retrain is required

The partition the current results came from is contaminated:

| Finding | Value |
| :--- | :---: |
| Records vs unique image hashes | 3,662 vs **3,534** |
| Duplicate groups spanning partitions | **48** |
| Test images byte-identical to a training image | 27 |
| Validation images byte-identical to a training image | 17 |
| Test images byte-identical to a validation image | 6 |
| Duplicate groups with conflicting severity labels | **30** |

Comparing scores on the contaminated test images against the rest does **not**
repair this. That comparison addresses test contamination only. It cannot
detect the effect of **validation** contamination, which influences which
epoch's checkpoint is selected, nor of **conflicting labels**, which place
contradictory supervision into training. Both require a clean retrain.

## What the corrected split does

[`build_clean_split.py`](../../backend/scripts/build_clean_split.py):

1. Hashes every image by its actual bytes.
2. Groups exact duplicates by that hash.
3. **Excludes** groups whose members disagree on the label — 30 groups, 62 images. They are not adjudicable from the data, and guessing would be fabrication.
4. Keeps one representative per surviving group.
5. Stratifies the unique images 70/15/15 by ICDR grade.
6. **Asserts zero hash overlap** between partitions and aborts if violated.

Expected output: **3,504 unique, non-conflicting images** — train 2,453 / val 526 / test 525.

| Grade | Eligible unique images |
| :--- | ---: |
| 0 — No DR | 1,796 |
| 1 — Mild NPDR | 338 |
| 2 — Moderate NPDR | 922 |
| 3 — Severe NPDR | 177 |
| 4 — Proliferative DR | 271 |
| **Total** | **3,504** |

---

## Resource configuration

| Setting | Value | Why |
| :--- | :--- | :--- |
| **Runtime type** | **GPU — T4** | Training needs CUDA. T4 matches the previous run, so epoch times stay comparable. A100/L4 also work and are faster. |
| RAM | Standard is sufficient | Peak usage is the DataLoader, not the model. High-RAM does no harm. |
| Disk | **~25 GB free** | 9.51 GB archive + ~10 GB extracted, with the archive deleted after unzip. |
| Wall-clock | **~80–95 min** | download ~7, hashing ~3, training ~50, evaluation ~2, validation evidence ~1, gate verification ~13, benchmark ~2 |

**Do everything in ONE GPU session.** Two reasons, both learned the hard way:

1. **Switching runtime type wipes the VM**, forcing a second 9.51 GB download.
2. **Latency figures from different Colab sessions are not comparable.** The previous benchmark appeared 1.91× faster than its baseline, but the two runs landed on different CPUs — stages whose code never changed were themselves 1.14–1.40× faster. Measuring the model and the latency on one machine removes that confound entirely.

The CPU benchmark runs fine inside a GPU session: it clears `CUDA_VISIBLE_DEVICES` before importing torch, so it measures that VM's CPU regardless of the accelerator attached.

---

## Run it

### Cell 1 — Kaggle token
```python
from google.colab import files
import os
uploaded = files.upload()                       # kaggle.json
os.makedirs(os.path.expanduser("~/.kaggle"), exist_ok=True)
with open(os.path.expanduser("~/.kaggle/kaggle.json"), "wb") as f:
    f.write(uploaded["kaggle.json"])
os.chmod(os.path.expanduser("~/.kaggle/kaggle.json"), 0o600)
```

### Cell 2 — fresh clone and dataset
```python
%cd /content
!rm -rf diabetic-retinopathy-cdss
!git clone -q https://github.com/CloudTechy/diabetic-retinopathy-cdss.git
%cd diabetic-retinopathy-cdss
!pip install -q pydantic-settings

!kaggle competitions download -c aptos2019-blindness-detection -p aptos2019
!unzip -q aptos2019/aptos2019-blindness-detection.zip -d aptos2019
!rm aptos2019/aptos2019-blindness-detection.zip
!nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
!ls aptos2019/train_images | wc -l          # expect 3662
```

Clone fresh rather than `git pull` — an older checkout will not contain the
clean-split builder.

### Cell 3 — inspect the split BEFORE training on it
```python
!python backend/scripts/build_clean_split.py aptos2019/train_images --out output/dataset_split_manifest.csv
```

Stop and report if these three lines do not appear:

```text
EXCLUDED 30 duplicate groups with conflicting labels (62 images)
Unique, non-conflicting images retained: 3504
Hash overlap between partitions: 0 (verified)
```

The script aborts on leakage rather than proceeding, so a silent pass is not possible.

### Cell 4 — retrain and regenerate (~50 min)
```python
!python notebooks/colab_train_and_evaluate.py
```

It reuses the already-downloaded images and calls the same clean split, so the
model trains on exactly the partition Cell 3 printed.

### Cell 5 — validation-gate evidence and decision preservation
```python
!python backend/scripts/generate_validation_evidence.py aptos2019/train_images
!python backend/scripts/verify_gate_downsampling.py aptos2019/train_images
```

`verify_gate_downsampling` must report **0 verdict changes** and a Laplacian
deviation of **exactly 0.0**.

### Cell 6 — CPU latency against the NEW checkpoint
```python
!python backend/scripts/benchmark_cpu_end_to_end.py --images-dir aptos2019/train_images --checkpoint output/efficientnet_b0_dr.pth --runs 30
```

`--checkpoint` is required: the freshly trained weights are in `output/`, not
yet installed at the repository path.

### Cell 7 — collect everything
```python
import shutil, os
os.makedirs('handover', exist_ok=True)
for f in ['dataset_split_manifest.csv','dataset_split_audit.json','efficientnet_b0_dr.pth',
          'training_execution.log','epoch_history.csv','training_summary.json',
          'held_out_predictions.csv','evaluation_summary.json',
          'benchmark_timings.csv','benchmark_summary.json',
          'learning_curves.png','confusion_matrix.png']:
    p = os.path.join('output', f)
    if os.path.exists(p): shutil.copy(p, 'handover/')
for f in ['validation_test_results.csv','gate_downsampling_verification.json',
          'cpu_end_to_end_benchmark.json','cpu_end_to_end_benchmark.csv']:
    p = os.path.join('docs/chapter4', f)
    if os.path.exists(p): shutil.copy(p, 'handover/')

print(sorted(os.listdir('handover')))
shutil.make_archive('clean_rerun_evidence', 'zip', 'handover')

from google.colab import files
files.download('clean_rerun_evidence.zip')
```

---

## After the run: the checkpoint digest will not match, and that is correct

The new weights have a new SHA-256. Until `MODEL_CHECKPOINT_SHA256` in
`backend/app/core/config.py` is updated to it, the integrity gate will fail on
`test_repository_checkpoint_matches_declared_digest`.

**That is the gate working.** It is refusing to let a checkpoint be served that
is not the one the committed evidence describes. The digest is updated as part
of installing the new artefacts, not before.

---

## Expect the numbers to move — and to be lower

Removing 158 images and eliminating leakage will most likely reduce reported
performance. **That is the correct outcome**, and it should be reported as-is.
A clean split that scores lower than a contaminated one is evidence the
contamination was inflating the result, which is precisely what the audit
predicted.

Do not tune hyperparameters to recover the previous figures. The run is
seeded and specified; report what it gives.

## After the run

Every document quoting a metric carries a superseded-results banner. Once the
clean artefacts are in, those banners come off and the figures are replaced
from the new run — no figure is edited by hand.

The integrity gate (`python backend/scripts/integrity_gate.py`) enforces that
the reported confusion matrix, accuracy and QWK recompute from the new
predictions file, so a mismatch fails the build rather than reaching Chapter
Four.
