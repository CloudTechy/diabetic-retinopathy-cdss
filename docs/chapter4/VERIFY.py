#!/usr/bin/env python3
"""
Independent verification of this evidence package.

    python VERIFY.py

NO INSTALLATION. No virtual environment, no PyTorch, no repository checkout, no
network. Python 3.8+ and the standard library only. Run it from the directory
this file sits in, after extracting the archive.

WHAT IT CHECKS, AND WHY EACH ONE MATTERS

  1. The checkpoint is the evaluated one.      A different checkpoint would make
                                               every reported metric describe a
                                               model nobody tested.
  2. The reported metrics recompute from the   Quadratic Weighted Kappa is
     raw predictions.                          implemented here from first
                                               principles, not imported, so this
                                               does not take the project's word
                                               for its own arithmetic.
  3. The integration fixture is a genuine      A pinned hash proves only that a
     held-out image.                           file has not changed. This checks
                                               the file IS the manifest row it
                                               names, and that the row is
                                               split=test.
  4. The split is leakage-free.                No held-out image may share its
                                               bytes with a training image.
  5. The test log records a passing run and    Five different test counts once
     EVERY document quotes THAT log.           circulated here. This reads every
                                               .md in the archive, including the
                                               root README, and recognises both
                                               "186 passed" and "Passed: 186".
  6. Every evidence file has a producer.       An artefact nobody claims to have
                                               produced is indistinguishable
                                               from a fabricated one.

Each check prints PASS or FAIL with the numbers it used, so a disagreement is
visible rather than asserted. The exit code is 0 only if all of them pass.

This script does not run the model. Doing so needs PyTorch and the test suite;
`docs/chapter4/reproducibility_runbook.md` covers that. Everything here is
arithmetic over files in this archive.
"""

import csv
import hashlib
import json
import os
import re
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))

# The archive is repository-relative: every file sits at the path it has in
# the project repository. Only this script, README.md and VERIFICATION.md are
# lifted to the root.
DOCS = os.path.join(HERE, "docs", "chapter4")
CHECKPOINT = os.path.join(HERE, "backend", "models", "weights", "efficientnet_b0_dr.pth")
MANIFEST = os.path.join(DOCS, "dataset_split_manifest.csv")
PREDICTIONS = os.path.join(DOCS, "held_out_predictions.csv")
METRICS = os.path.join(DOCS, "clinical_metrics.json")
TEST_LOG = os.path.join(DOCS, "test_execution.log")
PROVENANCE = os.path.join(DOCS, "evidence_provenance.md")

DECLARED_CHECKPOINT_SHA256 = (
    "67d0b89641f08057126dd411e380b25575ef29f71ae37ee5796d472d9203dbf7")

FIXTURE_ID = "d1f1ea894da1"
FIXTURE = os.path.join(HERE, "backend", "tests", "fixtures",
                       "aptos_heldout_%s.png" % FIXTURE_ID)

results = []


def check(name):
    def wrap(fn):
        results.append((name, fn))
        return fn
    return wrap


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_text(path):
    with open(path, encoding="utf-8", errors="replace") as fh:
        return fh.read()


def load_manifest():
    with open(MANIFEST, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


# ---------------------------------------------------------------------------
# 1. Checkpoint identity
# ---------------------------------------------------------------------------
@check("Checkpoint is the evaluated one")
def _checkpoint(out):
    if not os.path.exists(CHECKPOINT):
        return out("backend/models/weights/efficientnet_b0_dr.pth is missing"), False
    actual = sha256(CHECKPOINT)
    out("declared SHA-256 %s" % DECLARED_CHECKPOINT_SHA256)
    out("file     SHA-256 %s" % actual)
    out("size %s bytes" % format(os.path.getsize(CHECKPOINT), ","))
    return None, actual == DECLARED_CHECKPOINT_SHA256


# ---------------------------------------------------------------------------
# 2. Metrics recompute from the raw predictions
# ---------------------------------------------------------------------------
def quadratic_weighted_kappa(y_true, y_pred, n_classes=5):
    """Implemented from first principles so this is not a circular check."""
    n = len(y_true)
    observed = [[0] * n_classes for _ in range(n_classes)]
    for t, p in zip(y_true, y_pred):
        observed[t][p] += 1

    hist_t = [0] * n_classes
    hist_p = [0] * n_classes
    for t, p in zip(y_true, y_pred):
        hist_t[t] += 1
        hist_p[p] += 1

    denom = (n_classes - 1) ** 2
    numerator = 0.0
    denominator = 0.0
    for i in range(n_classes):
        for j in range(n_classes):
            w = ((i - j) ** 2) / denom
            expected = hist_t[i] * hist_p[j] / n
            numerator += w * observed[i][j]
            denominator += w * expected
    return 1.0 - numerator / denominator if denominator else 0.0


@check("Reported metrics recompute from held_out_predictions.csv")
def _metrics(out):
    with open(PREDICTIONS, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    y_true = [int(r["true_grade"]) for r in rows]
    y_pred = [int(r["predicted_grade"]) for r in rows]

    n = len(rows)
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    accuracy = 100.0 * correct / n
    kappa = quadratic_weighted_kappa(y_true, y_pred)

    with open(METRICS, encoding="utf-8") as fh:
        reported = json.load(fh)

    rep_n = reported.get("n") or reported.get("cohort_size") or reported.get("n_test")
    rep_acc = (reported.get("exact_accuracy_pct")
               or reported.get("accuracy_pct") or reported.get("accuracy"))
    rep_qwk = reported.get("qwk") or reported.get("quadratic_weighted_kappa")

    out("recomputed from %d raw predictions" % n)
    out("  accuracy  %.2f%%   reported %s" % (accuracy, rep_acc))
    out("  QWK       %.6f   reported %s" % (kappa, rep_qwk))
    out("  correct   %d / %d" % (correct, n))

    ok = True
    if rep_n is not None and int(rep_n) != n:
        out("  COHORT SIZE DISAGREES"); ok = False
    if rep_acc is not None and abs(float(rep_acc) - accuracy) > 0.011:
        out("  ACCURACY DISAGREES"); ok = False
    if rep_qwk is not None and abs(float(rep_qwk) - kappa) > 0.0001:
        out("  QWK DISAGREES"); ok = False
    return None, ok


# ---------------------------------------------------------------------------
# 3. The integration fixture is a genuine held-out image
# ---------------------------------------------------------------------------
@check("Integration fixture is a genuine held-out APTOS image")
def _fixture(out):
    if not os.path.exists(FIXTURE):
        out("fixture image absent from the package"); return None, False

    rows = {r["image_id"]: r for r in load_manifest()}
    record = rows.get(FIXTURE_ID)
    if record is None:
        out("%s does not appear in the manifest" % FIXTURE_ID); return None, False

    actual = sha256(FIXTURE)
    out("image_id      %s" % FIXTURE_ID)
    out("manifest split %s   (must be 'test')" % record["split"])
    out("manifest grade %s" % record["true_grade"])
    out("manifest SHA   %s" % record["sha256_hash"])
    out("file     SHA   %s" % actual)

    ok = record["split"] == "test" and record["sha256_hash"] == actual
    if record["split"] != "test":
        out("  NOT HELD OUT - grading a training image proves nothing")
    if record["sha256_hash"] != actual:
        out("  FILE IS NOT THE IMAGE ITS NAME CLAIMS")
    return None, ok


# ---------------------------------------------------------------------------
# 4. The split is leakage-free
# ---------------------------------------------------------------------------
@check("Held-out split shares no image bytes with training")
def _leakage(out):
    rows = load_manifest()
    by_split = {}
    for r in rows:
        by_split.setdefault(r["split"], set()).add(r["sha256_hash"])

    counts = Counter(r["split"] for r in rows)
    out("manifest rows %d   %s" % (len(rows), dict(counts)))

    train = by_split.get("train", set())
    test = by_split.get("test", set())
    val = by_split.get("val", set())

    leaked_test = len(train & test)
    leaked_val = len(train & val)
    out("held-out images byte-identical to a training image: %d" % leaked_test)
    out("validation images byte-identical to a training image: %d" % leaked_val)

    groups = Counter(r.get("duplicate_group_id", "") for r in rows)
    multi = sum(1 for c in groups.values() if c > 1)
    out("duplicate groups %d, of which %d hold more than one row"
        % (len(groups), multi))

    return None, leaked_test == 0 and leaked_val == 0


# ---------------------------------------------------------------------------
# 5. The test log passes, and the documents quote THAT log
# ---------------------------------------------------------------------------
@check("Documents quote the committed test log")
def _testlog(out):
    text = read_text(TEST_LOG)

    collected = re.findall(r"collected (\d+) item", text)
    summary = re.findall(
        r"(\d+) passed(?:, (\d+) failed)?(?:, (\d+) skipped)?", text)
    if not summary:
        out("no pytest summary in the log"); return None, False

    passed = int(summary[-1][0])
    failed = int(summary[-1][1] or 0)
    skipped = int(summary[-1][2] or 0)
    total = int(collected[-1]) if collected else passed + skipped

    out("log records: %d collected, %d passed, %d failed, %d skipped"
        % (total, passed, failed, skipped))

    ok = failed == 0 and passed + skipped == total
    if failed:
        out("  THE LOG RECORDS FAILURES")
    if passed + skipped != total:
        out("  THE LOG'S OWN ARITHMETIC DOES NOT CLOSE")

    allowed = {"collected": total, "passed": passed,
               "skipped": skipped, "failed": failed}

    # Two wordings occur in this package. An earlier version of this scanner
    # recognised only the first and read only documentation/, so "Passed: 182",
    # "Tests Collected: 186", "181/181", and the root README.md and
    # REVIEWER_RESPONSE.md were all invisible to it - and it reported agreement
    # while four counts disagreed.
    #
    # The negative lookbehinds keep "Gate 1 Passed" and "All 3 Passed" out:
    # those are gate verdicts, not suite counts.
    patterns = [
        re.compile(r"(?<!Gate )(?<!All )\*{0,2}(\d+)\*{0,2}\s*"
                   r"(?:tests?\s+(?:cases?\s+)?)?"
                   r"(collected|passed|skipped|failed)", re.I),
        re.compile(r"(collected|passed|skipped|failed)\s*:\s*\*{0,2}(\d+)", re.I),
    ]

    # Every Markdown file in the package. archive/ is excluded by name: a
    # superseded document's purpose is to record what WAS true.
    targets = []
    for base, _dirs, files in os.walk(HERE):
        parts = base.replace("\\", "/").split("/")
        if "archive" in parts:
            continue
        for name in sorted(files):
            if name.endswith(".md"):
                targets.append(os.path.join(base, name))

    disagreements = []
    for path in sorted(targets):
        rel = os.path.relpath(path, HERE)
        for lineno, line in enumerate(read_text(path).split("\n"), 1):
            found = [(k.lower(), int(v)) for v, k in patterns[0].findall(line)]
            found += [(k.lower(), int(v)) for k, v in patterns[1].findall(line)]
            for kind, value in found:
                if kind in allowed and value != allowed[kind]:
                    disagreements.append("%s:%d says %s %s"
                                         % (rel, lineno, value, kind))

    if disagreements:
        ok = False
        out("  DOCUMENTS DISAGREE WITH THE LOG:")
        for d in disagreements[:8]:
            out("    %s" % d)
    else:
        out("every test count in every .md in this package matches this log")
    return None, ok


# ---------------------------------------------------------------------------
# 6. Every evidence file has a declared producer
# ---------------------------------------------------------------------------
@check("Every evidence file names the script that produced it")
def _provenance(out):
    if not os.path.exists(PROVENANCE):
        out("docs/chapter4/evidence_provenance.md is missing"); return None, False
    text = read_text(PROVENANCE)

    # Evidence is data: anything under docs/chapter4/ that is not prose or
    # code, plus the trained weights. Each must be named in the provenance
    # document, which says what produced it.
    EVIDENCE = (".csv", ".json", ".log", ".png", ".txt", ".pth")
    undeclared = []
    candidates = []
    if os.path.isdir(DOCS):
        candidates += [os.path.join(DOCS, n) for n in sorted(os.listdir(DOCS))]
    candidates.append(CHECKPOINT)
    for path in candidates:
        name = os.path.basename(path)
        if not os.path.isfile(path) or name.startswith("."):
            continue
        if not name.endswith(EVIDENCE):
            continue
        if name not in text:
            undeclared.append(os.path.relpath(path, HERE).replace("\\", "/"))

    if undeclared:
        out("files with no entry in PROVENANCE.md:")
        for u in undeclared:
            out("  %s" % u)
        return None, False

    out("every evidence file under docs/chapter4/ and the checkpoint are declared")
    return None, True


# ---------------------------------------------------------------------------
def main():
    print("=" * 72)
    print("INDEPENDENT VERIFICATION OF THE CHAPTER 4 EVIDENCE PACKAGE")
    print("=" * 72)
    print("Python %s" % sys.version.split()[0])
    print("Standard library only. No model is run; this is arithmetic over the")
    print("files in this archive.")
    print()

    failures = 0
    for index, (name, fn) in enumerate(results, 1):
        lines = []
        print("[%d] %s" % (index, name))
        try:
            _, ok = fn(lines.append)
        except Exception as exc:                      # noqa: BLE001
            lines.append("raised %s: %s" % (type(exc).__name__, exc))
            ok = False
        for line in lines:
            print("      %s" % line)
        print("      -> %s" % ("PASS" if ok else "FAIL"))
        print()
        if not ok:
            failures += 1

    print("=" * 72)
    if failures:
        print("%d of %d checks FAILED." % (failures, len(results)))
        print("Each failure above prints the numbers it used. A disagreement is")
        print("a defect in this package, not in your copy of it; please report")
        print("it with the output above.")
    else:
        print("All %d checks passed." % len(results))
        print()
        print("This establishes that the package is internally consistent and")
        print("that its headline metrics recompute from the raw predictions.")
        print("It does NOT establish clinical validity, generalisation beyond")
        print("APTOS 2019, or fitness for clinical use, and no such claim is")
        print("made anywhere in this package.")
    print("=" * 72)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
