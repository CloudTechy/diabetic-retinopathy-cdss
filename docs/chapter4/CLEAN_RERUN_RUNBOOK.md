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

## Run it

**Colab, GPU runtime (T4).** ~1.5 hours end to end.

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

### Cell 2 — repo and dataset
```python
!git clone -q https://github.com/CloudTechy/diabetic-retinopathy-cdss.git
%cd diabetic-retinopathy-cdss
!pip install -q pydantic-settings

!kaggle competitions download -c aptos2019-blindness-detection -p aptos2019
!unzip -q aptos2019/aptos2019-blindness-detection.zip -d aptos2019
!rm aptos2019/aptos2019-blindness-detection.zip
!ls aptos2019/train_images | wc -l          # expect 3662
```

### Cell 3 — verify the split before training on it
```python
!python backend/scripts/build_clean_split.py aptos2019/train_images \
    --out output/dataset_split_manifest.csv
```

Confirm before continuing:
- `Unique, non-conflicting images retained: 3504`
- `EXCLUDED 30 duplicate groups with conflicting labels`
- `Hash overlap between partitions: 0 (verified)`

If any differ, **stop** and report them. The script aborts on leakage rather
than proceeding.

### Cell 4 — retrain and regenerate everything
```python
!python notebooks/colab_train_and_evaluate.py
```

Uses the same clean split. Produces the checkpoint, training transcript, epoch
history, held-out predictions, evaluation summary, benchmark and both plots.

### Cell 5 — validation-gate evidence and the decision-preservation check
```python
!python backend/scripts/generate_validation_evidence.py aptos2019/train_images
!python backend/scripts/verify_gate_downsampling.py aptos2019/train_images
```

### Cell 6 — CPU benchmark
Switch to a **CPU** runtime, re-run Cells 2–3, then:
```python
!python backend/scripts/benchmark_cpu_end_to_end.py \
    --images-dir aptos2019/train_images --runs 30
```

### Cell 7 — download
```python
from google.colab import files
import shutil
shutil.make_archive('clean_rerun_evidence', 'zip', 'output')
files.download('clean_rerun_evidence.zip')
files.download('docs/chapter4/validation_test_results.csv')
files.download('docs/chapter4/gate_downsampling_verification.json')
files.download('docs/chapter4/cpu_end_to_end_benchmark.json')
```

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
