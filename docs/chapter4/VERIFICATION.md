# Independent Verification

**No virtual environment, no PyTorch, no repository checkout, no network.**
Extract this archive, open a terminal in this folder, and run:

```bash
python VERIFY.py
```

Python 3.8 or later, standard library only. It finishes in seconds.

---

## Why this exists

A reviewer should not have to take a submission's word for its own arithmetic.
`VERIFY.py` recomputes the headline figures from the raw predictions, with
Quadratic Weighted Kappa **implemented from first principles rather than
imported**, so agreement is evidence rather than circularity.

Every check prints the numbers it used. If one fails on your copy, the output
shows exactly which figures disagree — you do not have to read any code to see
where the problem is.

---

## What it checks

| # | Check | What a failure would mean |
| :-- | :--- | :--- |
| 1 | The checkpoint is the evaluated one | Every reported metric would describe a model nobody tested |
| 2 | Metrics recompute from `held_out_predictions.csv` | The reported accuracy or kappa is not what the predictions support |
| 3 | The integration fixture is a genuine held-out image | The end-to-end test would be grading something other than a held-out APTOS photograph |
| 4 | The split is leakage-free | Test performance would be partly memorisation |
| 5 | Documents quote the committed test log | The suite result stated in prose would not be a run that happened |
| 6 | Every evidence file has a declared producer | An artefact nobody claims to have produced is indistinguishable from a fabricated one |

Checks 3, 5 and 6 exist because each one caught a real defect in an earlier
revision of this package. They are not decoration.

---

## What it does NOT check

It does not run the model, the validation gates or the test suite: those need
PyTorch, and [`docs/chapter4/reproducibility_runbook.md`](docs/chapter4/reproducibility_runbook.md)
covers them.

It does not establish clinical validity, generalisation beyond APTOS 2019, or
fitness for clinical use. **No such claim is made anywhere in this package.**

---

## Recorded transcript

Produced on 2026-10-03 from this package. Your run should match it line for line.

```text
========================================================================
INDEPENDENT VERIFICATION OF THE CHAPTER 4 EVIDENCE PACKAGE
========================================================================
Python 3.13.5
Standard library only. No model is run; this is arithmetic over the
files in this archive.

[1] Checkpoint is the evaluated one
      declared SHA-256 67d0b89641f08057126dd411e380b25575ef29f71ae37ee5796d472d9203dbf7
      file     SHA-256 67d0b89641f08057126dd411e380b25575ef29f71ae37ee5796d472d9203dbf7
      size 16,358,249 bytes
      config.py              67d0b89641f08057... agrees
      training_summary.json  67d0b89641f08057... agrees
      -> PASS

[2] Reported metrics recompute from held_out_predictions.csv
      recomputed from 525 raw predictions
        accuracy  84.00%   reported 84.0
        QWK       0.865832   reported 0.8658318425760286
        correct   441 / 525
      prediction ids == manifest test split: yes (525 vs 525)
      README headline table quotes the recomputed values: yes
      -> PASS

[3] Integration fixture is a genuine held-out APTOS image
      image_id      d1f1ea894da1
      manifest split test   (must be 'test')
      manifest grade 2
      manifest SHA   d6eb606b07cdcd6045f2477f9fa3899df7f112f862897286a21f0eece06f2acf
      file     SHA   d6eb606b07cdcd6045f2477f9fa3899df7f112f862897286a21f0eece06f2acf
      -> PASS

[4] Held-out split shares no image bytes with training
      manifest rows 3504   {'test': 525, 'train': 2453, 'val': 526}
      held-out images byte-identical to a training image: 0
      validation images byte-identical to a training image: 0
      held-out images byte-identical to a validation image: 0
      duplicate groups 3504, of which 0 hold more than one row
      hashes appearing in more than one split: 0
      -> PASS

[5] Documents quote the committed test log
      log records: 218 collected, 217 passed, 0 failed, 1 skipped
      every test count in every .md in this package matches this log
      -> PASS

[6] Every evidence file names the script that produced it
      every evidence file under docs/chapter4/ and the checkpoint are declared
      -> PASS

========================================================================
All 6 checks passed.

This establishes that the package is internally consistent and
that its headline metrics recompute from the raw predictions.
It does NOT establish clinical validity, generalisation beyond
APTOS 2019, or fitness for clinical use, and no such claim is
made anywhere in this package.
========================================================================
```

---

## Going further: the manifest against the real corpus

Check 3 proves the integration fixture is the manifest row it names. To verify
the **manifest itself** against your own Kaggle download of APTOS 2019:

```bash
python backend/scripts/corpus_guard.py <your-path>/aptos2019/train_images --full
```

It hashes every file and compares against `dataset_split_manifest.csv`, and
refuses any file that carries a manifest `image_id` without matching its
recorded SHA-256. That check was added after fifteen placeholder images were
found in a working directory wearing real held-out image identifiers.

---

## If a check fails

Report it with the output. A disagreement is a defect in this package, not in
your copy of it.
