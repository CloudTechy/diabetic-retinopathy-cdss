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
    ok = actual == DECLARED_CHECKPOINT_SHA256

    # The constant above is one claim. The system's own config and the training
    # run's summary are two more; all three must name the same bytes, so this
    # check cannot be satisfied by editing one file.
    config = os.path.join(HERE, "backend", "app", "core", "config.py")
    summary = os.path.join(DOCS, "training_summary.json")
    for label, path, pattern in (
            ("config.py", config, r"MODEL_CHECKPOINT_SHA256.{0,80}?([0-9a-f]{64})"),
            ("training_summary.json", summary, r"\"checkpoint_sha256\"\s*:\s*\"([0-9a-f]{64})\"")):
        if not os.path.exists(path):
            out("%s is missing" % label); ok = False; continue
        m = re.search(pattern, read_text(path), re.S)
        if not m:
            out("%s does not state a checkpoint digest" % label); ok = False; continue
        agree = m.group(1) == actual
        out("%-22s %s %s" % (label, m.group(1)[:16] + "...", "agrees" if agree else "DISAGREES"))
        ok = ok and agree
    return None, ok


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
    if rep_n is None or rep_acc is None or rep_qwk is None:
        out("  clinical_metrics.json lacks n_test / exact_accuracy_pct / quadratic_weighted_kappa"); ok = False
    if rep_n is not None and int(rep_n) != n:
        out("  COHORT SIZE DISAGREES"); ok = False
    if rep_acc is not None and abs(float(rep_acc) - accuracy) > 0.011:
        out("  ACCURACY DISAGREES"); ok = False
    if rep_qwk is not None and abs(float(rep_qwk) - kappa) > 0.0001:
        out("  QWK DISAGREES"); ok = False

    # The predictions must be the manifest's held-out split - every id, and
    # only those ids - with the manifest's own grade as the truth column.
    manifest = {r["image_id"]: r for r in load_manifest()}
    test_ids = {i for i, r in manifest.items() if r["split"] == "test"}
    pred_ids = {r["image_id"] for r in rows}
    out("prediction ids == manifest test split: %s (%d vs %d)"
        % ("yes" if pred_ids == test_ids else "NO", len(pred_ids), len(test_ids)))
    if pred_ids != test_ids:
        ok = False
    mislabelled = sum(1 for r in rows if r["image_id"] in manifest
                      and str(manifest[r["image_id"]]["true_grade"]) != str(r["true_grade"]))
    if mislabelled:
        out("  %d predictions carry a truth grade different from the manifest" % mislabelled); ok = False

    # The headline table every reader opens first must quote these numbers.
    readme = os.path.join(HERE, "README.md")
    if os.path.exists(readme):
        text = read_text(readme)
        within = 100.0 * sum(1 for t, p in zip(y_true, y_pred) if abs(t - p) <= 1) / n
        wanted = {
            "Quadratic Weighted Kappa": "%.4f" % kappa,
            "Exact accuracy": "%.2f%%" % accuracy,
            "Within-one-grade agreement": "%.2f%%" % within,
            "Held-out cohort": "N = %d" % n,
        }
        # Operating points, recomputed here from the predictions.
        def op(threshold):
            tp = sum(1 for t, p in zip(y_true, y_pred) if t >= threshold and p >= threshold)
            fn = sum(1 for t, p in zip(y_true, y_pred) if t >= threshold and p < threshold)
            fp = sum(1 for t, p in zip(y_true, y_pred) if t < threshold and p >= threshold)
            tn = sum(1 for t, p in zip(y_true, y_pred) if t < threshold and p < threshold)
            return (100.0 * tp / (tp + fn), 100.0 * tn / (tn + fp), 100.0 * tn / (tn + fn))
        r_sens, r_spec, _ = op(2)
        s_sens, _, s_npv = op(3)
        wanted_many = {
            "Referable DR": ["%.1f%%" % r_sens, "%.1f%%" % r_spec],
            "Sight-threatening DR": ["%.1f%%" % s_sens, "%.1f%%" % s_npv],
        }
        for label, token in wanted.items():
            m = re.search(r"^\|\s*\*\*%s\*\*[^|]*\|(.*)\|\s*$" % re.escape(label), text, re.M)
            if not m or token not in m.group(1):
                out("  README row **%s** does not state %s" % (label, token)); ok = False
        for label, tokens in wanted_many.items():
            m = re.search(r"^\|\s*\*\*%s\*\*[^|]*\|(.*)\|\s*$" % re.escape(label), text, re.M)
            for token in tokens:
                if not m or token not in m.group(1):
                    out("  README row **%s** does not state %s" % (label, token)); ok = False
        out("README headline table quotes the recomputed values: %s" % ("yes" if ok else "NO"))
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

    leaked_val_test = len(val & test)
    out("held-out images byte-identical to a validation image: %d" % leaked_val_test)

    groups = Counter(r.get("duplicate_group_id", "") for r in rows)
    multi = sum(1 for c in groups.values() if c > 1)
    out("duplicate groups %d, of which %d hold more than one row"
        % (len(groups), multi))

    splits_per_hash = {}
    for r in rows:
        splits_per_hash.setdefault(r["sha256_hash"], set()).add(r["split"])
    spanning = sum(1 for v in splits_per_hash.values() if len(v) > 1)
    out("hashes appearing in more than one split: %d" % spanning)

    # Each of these was printed and ignored before; each is now a failure.
    ok = (leaked_test == 0 and leaked_val == 0 and leaked_val_test == 0
          and multi == 0 and spanning == 0)
    if not ok:
        out("  THE SPLIT IS NOT LEAKAGE-FREE")
    return None, ok


# ---------------------------------------------------------------------------
# 5. The test log passes, and the documents quote THAT log
# ---------------------------------------------------------------------------
@check("Documents quote the committed test log")
def _testlog(out):
    text = read_text(TEST_LOG)

    collected = re.findall(r"collected (\d+) item", text)
    # pytest prints "N failed, M passed, K skipped" in ITS order; parse each
    # kind independently from the final summary line rather than assuming one.
    final = [ln for ln in text.strip().split("\n") if re.search(r"\d+ passed|\d+ failed", ln)]
    if not final:
        out("no pytest summary in the log"); return None, False
    last = final[-1]

    def count(kind):
        m = re.search(r"(\d+) %s" % kind, last)
        return int(m.group(1)) if m else 0

    passed, failed, skipped = count("passed"), count("failed"), count("skipped")
    errors = count("error")
    total = int(collected[-1]) if collected else passed + skipped + failed
    if errors:
        out("  THE LOG RECORDS %d ERROR(S)" % errors)
        failed += errors

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
        re.compile(r"(?<!Gate )\*{0,2}(\d+)\*{0,2}\s*"
                   r"(?:tests?\s+(?:cases?\s+)?)?"
                   r"(collected|passed|skipped|failed)", re.I),
        re.compile(r"(collected|passed|skipped|failed)\s*:\s*\*{0,2}(\d+)", re.I),
    ]
    # "194/195" and "194 of 195" state passed and collected at once; "195 tests"
    # states collected. "All 3 Passed" in a gate table is excluded by requiring
    # the number to be at least the suite's size order (a gate table says 3).
    extra = [
        (re.compile(r"\b(\d+)\s*(?:/|of)\s*(\d+)\s*(?:tests?\s+)?passed", re.I), ("passed", "collected")),
        (re.compile(r"\b(\d+)\s+automated tests\b", re.I), ("collected",)),
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
            found = [(k.lower(), int(v)) for v, k in patterns[0].findall(line)
                     if not (int(v) < 20 and re.search(r"\bAll\s+%s\b" % v, line))]
            found += [(k.lower(), int(v)) for k, v in patterns[1].findall(line)]
            for pattern, kinds in extra:
                for m in pattern.finditer(line):
                    found += list(zip(kinds, (int(g) for g in m.groups())))
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
    # Every subfolder too. The superseded N=549 confusion-matrix screenshot sat
    # under screenshots/, which the earlier top-level listing never reached.
    for base, dirs, files in os.walk(DOCS):
        dirs[:] = sorted(d for d in dirs if d != "archive")
        candidates += [os.path.join(base, n) for n in sorted(files)]
    candidates.append(CHECKPOINT)

    # A producer cell of "none" or "produced none" declares nothing.
    declared_rows = {}
    for line in text.split("\n"):
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) >= 2 and cells[0].startswith("`"):
            declared_rows[cells[0].strip("`")] = cells[1]

    def is_declared(name):
        for key, producer in declared_rows.items():
            if name in key and producer and "none" not in producer.lower():
                return True
        return False

    # A named producer must be a script in this archive that mentions the
    # artefact it is said to write - a static check, but one a wrong name
    # cannot pass. Rows whose producer is a notebook, a shell command or a
    # recorded git commit are checked where a script path is given.
    wrong_producer = []
    for key, producer in declared_rows.items():
        base = os.path.basename(key)
        if not base.endswith(EVIDENCE) or "none" in producer.lower():
            continue
        m = re.search(r"`((?:backend/scripts|notebooks)/[\w./-]+\.py)`", producer)
        if not m:
            continue
        script = os.path.join(HERE, m.group(1))
        if not os.path.exists(script):
            wrong_producer.append("%s: producer %s is not in the archive" % (key, m.group(1)))
            continue
        stem = base.rsplit(".", 1)[0]
        stem = re.sub(r"_\d{4}-\d{2}-\d{2}_.*$", "", stem)   # benchmark_history dated copies
        stem = re.sub(r"\.superseded_[0-9a-f]+$", "", stem)   # archived copies renamed by commit
        if stem not in read_text(script) and base not in read_text(script):
            wrong_producer.append("%s: %s never mentions it" % (key, m.group(1)))
    for w in wrong_producer:
        out("  " + w)
    if wrong_producer:
        return None, False

    for path in candidates:
        name = os.path.basename(path)
        if not os.path.isfile(path) or name.startswith("."):
            continue
        if not name.endswith(EVIDENCE):
            continue
        rel = os.path.relpath(path, HERE).replace("\\", "/")
        # screenshots are declared as a range, "01`-`08"
        if "/screenshots/" in rel and re.match(r"0[1-8]", name) and is_declared("screenshots/01"):
            continue
        if not is_declared(name):
            undeclared.append(rel)

    # "copy of X" is a checkable claim: the bytes must be X's.
    copies_wrong = []
    for m in re.finditer(r"^\|\s*`([^`]+)`\s*\|\s*copy of\s*`([^`]+)`", text, re.M):
        a, b = os.path.join(HERE, m.group(1)), os.path.join(HERE, m.group(2))
        if not (os.path.exists(a) and os.path.exists(b)) or sha256(a) != sha256(b):
            copies_wrong.append("%s is not a byte copy of %s" % (m.group(1), m.group(2)))
    for w in copies_wrong:
        out("  " + w)
    if copies_wrong:
        return None, False

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
