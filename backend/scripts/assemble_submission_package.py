#!/usr/bin/env python3
"""
Build the Chapter 4 evidence archive directly from the repository tree.

    python backend/scripts/assemble_submission_package.py --check
    python backend/scripts/assemble_submission_package.py --list
    python backend/scripts/assemble_submission_package.py --build <out.zip>

LAYOUT. The archive is REPOSITORY-RELATIVE: every file ships at the path it
has in this repository, under one top-level folder. Three files are lifted to
the archive root because they are what a reviewer touches first:

    docs/chapter4/SUBMISSION_README.md  ->  README.md
    docs/chapter4/VERIFY.py             ->  VERIFY.py
    docs/chapter4/VERIFICATION.md       ->  VERIFICATION.md

WHY. Earlier builds re-homed every artefact into documentation/,
logs_and_metrics/, scripts/ and source/, and kept a staged copy of the result
under docs/chapter4_submission_package/. Three defects followed, and an
independent review found all three:

  - 94 relative links broke, because sibling files were split across folders.
  - The extracted test suite could not run: conftest imports `main`, and the
    staged copy had never carried main.py.
  - A corrected VERIFY.py was silently reverted by two scripts mirroring the
    staged copy in opposite directions.

With the repository layout preserved, links resolve by construction, the gate
rules resolve their paths unchanged whether run here or inside the extracted
archive, and there is exactly one copy of everything. The staged directory is
retired; `--check` fails if it reappears.

NOTHING IS GENERATED HERE. Every file is copied byte-for-byte. A missing
artefact fails the build; it is never substituted.

`--build` writes the archive, extracts it to a temporary directory, runs
VERIFY.py there, and keeps the archive only if every check passes - so the
thing that ships is the thing that was verified. The recorded transcript in
docs/chapter4/VERIFICATION.md is refreshed from that run.
"""

import argparse
import hashlib
import io
import os
import posixpath
import re
import subprocess
import sys
import tempfile
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))
CHAPTER4 = os.path.join(REPO_ROOT, "docs", "chapter4")

EXPECTED_CHECKPOINT_SHA256 = "67d0b89641f08057126dd411e380b25575ef29f71ae37ee5796d472d9203dbf7"
ARCHIVE_DIRNAME = "DR-CDSS_Chapter4_Evidence"

# Lifted to the archive root. Inside an extracted archive these sources no
# longer exist at docs/chapter4/, so manifest() takes them from the root.
ROOT_FILES = [
    ("docs/chapter4/SUBMISSION_README.md", "README.md"),
    ("docs/chapter4/VERIFY.py", "VERIFY.py"),
    ("docs/chapter4/VERIFICATION.md", "VERIFICATION.md"),
]

# Single files, shipped at their repository path.
SINGLE_FILES = [
    "backend/main.py",
    "backend/requirements.txt",
    "backend/pytest.ini",
    "backend/models/weights/efficientnet_b0_dr.pth",
    "notebooks/colab_train_and_evaluate.py",
    "PROGRESS_TRACKER.md",
    ".env.example",
    ".github/workflows/evidence-integrity-gate.yml",
    "frontend/scripts/capture_live_screenshots.js",
    "frontend/scripts/capture_rejection.js",
    "frontend/scripts/check_preflight.cjs",
    "frontend/scripts/fixtures/aptos_heldout_d1f1ea894da1_256.rgba",
    "frontend/scripts/fixtures/aptos_heldout_d1f1ea894da1_256.json",
    "docs/model_integration_guide.md",
    # training_environment.md sources its frontend and database versions from these.
    "frontend/package-lock.json",
    "docker-compose.yml",
    # Everything `docker compose up --build` and `npm ci && npm run build`
    # consume. The runbook advertises both; an archive that cannot do what
    # its runbook says is a defect, and a rule checks these against the
    # Dockerfiles and package.json.
    "backend/Dockerfile",
    "frontend/Dockerfile",
    "frontend/package.json",
    "frontend/index.html",
    "frontend/vite.config.ts",
    "frontend/tsconfig.json",
    "frontend/tsconfig.node.json",
    "frontend/tailwind.config.js",
    "frontend/postcss.config.js",
]

# Whole trees, shipped at their repository paths. A suffix tuple limits what
# is taken; None takes every file.
TREES = [
    ("backend/app", (".py",)),
    ("backend/tests", (".py", ".png")),
    ("backend/scripts", (".py",)),
    ("frontend/src", (".ts", ".tsx", ".css")),
    ("docs/chapter4", None),
]

# Directories directly under docs/chapter4 that are deliberately NOT shipped,
# each with the reason. --check reports anything under docs/chapter4 that is
# neither shipped nor listed here.
EXCLUDED_CHAPTER4_DIRS = {
    "screenshots_fixtures": "superseded fixture-based captures; replaced by "
                            "screenshots/, retained in the repository as a record",
}

SKIP_DIRS = {"__pycache__", ".pytest_cache", "node_modules", ".git"}
TEST_LOG_PATTERN = re.compile(r"test.*(execution|run|output)", re.I)
RETIRED_STAGING_DIR = os.path.join(REPO_ROOT, "docs", "chapter4_submission_package")

# Fixed so a rebuild of identical content is byte-identical.
ARCHIVE_TIMESTAMP = (2026, 10, 3, 0, 0, 0)


def _rel(path):
    return os.path.relpath(path, REPO_ROOT).replace("\\", "/")


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest():
    """
    [(source relative to the repository, destination relative to the archive
    root)], sorted by destination. Raises if two sources claim one destination.
    """
    entries = {}

    def put(src, dst):
        if dst in entries and entries[dst] != src:
            raise ValueError("two sources map to %s: %s and %s"
                             % (dst, entries[dst], src))
        entries[dst] = src

    lifted = set()
    for src, dst in ROOT_FILES:
        lifted.add(src)
        if os.path.exists(os.path.join(REPO_ROOT, src)):
            put(src, dst)
        else:
            put(dst, dst)

    for src in SINGLE_FILES:
        put(src, src)

    for tree, suffixes in TREES:
        root = os.path.join(REPO_ROOT, tree)
        if not os.path.isdir(root):
            continue
        for base, dirs, names in os.walk(root):
            rel_base = _rel(base)
            dirs[:] = sorted(
                d for d in dirs
                if d not in SKIP_DIRS
                and not (rel_base == "docs/chapter4" and d in EXCLUDED_CHAPTER4_DIRS))
            for name in sorted(names):
                if name.endswith((".pyc", ".pyo")):
                    continue
                if suffixes and not name.endswith(suffixes):
                    continue
                src = _rel(os.path.join(base, name))
                if src in lifted:
                    continue
                put(src, src)

    return sorted(((src, dst) for dst, src in entries.items()), key=lambda e: e[1])


def crlf_files(entries):
    """Text files of the manifest that contain a CR LF. The repository holds LF
    only, so a CR LF on disk is a file that differs from its own commit."""
    bad = []
    for src, _dst in entries:
        with open(os.path.join(REPO_ROOT, src), "rb") as fh:
            data = fh.read()
        if b"\0" not in data and b"\r\n" in data:
            bad.append(src)
    return bad


def problems():
    """Every reason the archive must not be built, as a list of strings."""
    found = []

    ckpt = os.path.join(REPO_ROOT, "backend", "models", "weights", "efficientnet_b0_dr.pth")
    if not os.path.exists(ckpt):
        found.append("checkpoint missing: backend/models/weights/efficientnet_b0_dr.pth")
    else:
        actual = sha256_file(ckpt)
        if actual != EXPECTED_CHECKPOINT_SHA256:
            found.append("checkpoint digest mismatch: expected %s, found %s - refusing "
                         "to package weights that are not the evaluated ones"
                         % (EXPECTED_CHECKPOINT_SHA256, actual))

    try:
        entries = manifest()
    except ValueError as exc:
        return found + [str(exc)]

    sources = {src for src, _ in entries}
    for src, dst in entries:
        if not os.path.exists(os.path.join(REPO_ROOT, src)):
            found.append("missing: %s (would ship as %s)" % (src, dst))

    # Completeness: nothing committed under docs/chapter4 may be silently
    # left out. Either it ships or it is in EXCLUDED_CHAPTER4_DIRS with a reason.
    lifted = {src for src, _ in ROOT_FILES}
    if os.path.isdir(CHAPTER4):
        for base, dirs, names in os.walk(CHAPTER4):
            rel_base = _rel(base)
            dirs[:] = sorted(
                d for d in dirs
                if d not in SKIP_DIRS
                and not (rel_base == "docs/chapter4" and d in EXCLUDED_CHAPTER4_DIRS))
            for name in names:
                if name.endswith((".pyc", ".pyo")):
                    continue
                rel = _rel(os.path.join(base, name))
                if rel not in sources and rel not in lifted:
                    found.append("under docs/chapter4 but not in the archive: %s" % rel)

    logs = [dst for _, dst in entries if TEST_LOG_PATTERN.search(posixpath.basename(dst))]
    if len(logs) != 1:
        found.append("exactly one test log must ship; found %d: %s" % (len(logs), logs))

    if os.path.isdir(RETIRED_STAGING_DIR):
        found.append("docs/chapter4_submission_package/ exists. The staged copy is "
                     "retired: the archive is built from the repository tree, and a "
                     "second copy is how a corrected file was silently reverted.")

    if not any("map to" in p for p in found):
        try:
            for src in crlf_files(manifest()):
                found.append("CR LF line endings (the commit holds LF): %s" % src)
        except OSError as exc:
            found.append(str(exc))
    return found


def check(verbose=True):
    found = problems()
    if verbose:
        entries = manifest() if not any("map to" in p for p in found) else []
        print("archive entries: %d" % len(entries))
        for p in found:
            print("  [PROBLEM] %s" % p)
    return not found


def _write_archive(entries):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for src, dst in entries:
            info = zipfile.ZipInfo(ARCHIVE_DIRNAME + "/" + dst, date_time=ARCHIVE_TIMESTAMP)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            with open(os.path.join(REPO_ROOT, src), "rb") as fh:
                zf.writestr(info, fh.read())
    return buf.getvalue()


def _verify_extracted(archive_bytes):
    """Extract to a temp dir and run VERIFY.py there. Returns (ok, transcript)."""
    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(io.BytesIO(archive_bytes)) as zf:
            zf.extractall(tmp)
        root = os.path.join(tmp, ARCHIVE_DIRNAME)
        proc = subprocess.run([sys.executable, "VERIFY.py"], cwd=root,
                              capture_output=True, text=True)
        transcript = (proc.stdout or "") + (proc.stderr or "")
        return proc.returncode == 0, transcript


TRANSCRIPT_BLOCK = re.compile(r"(## Recorded transcript\n.*?```text\n)(.*?)(\n```)", re.S)


def _refresh_transcript(transcript):
    """Splice VERIFY.py's output into docs/chapter4/VERIFICATION.md. True if changed."""
    path = os.path.join(CHAPTER4, "VERIFICATION.md")
    if not os.path.exists(path):
        return False
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    m = TRANSCRIPT_BLOCK.search(text)
    if not m:
        return False
    new = m.group(1) + transcript.strip("\n") + m.group(3)
    updated = text[:m.start()] + new + text[m.end():]
    # The date beside the transcript is the date of the recorded build check
    # (build_verification.log), a file in the archive - never the clock. An
    # earlier version wrote today's date here, so the same commit built on two
    # days gave two archives while the README said a rebuild was byte-identical.
    with open(os.path.join(CHAPTER4, "build_verification.log"), encoding="utf-8") as fh:
        checked = re.search(r"date \(UTC\): (\d{4}-\d{2}-\d{2})", fh.read())
    if checked:
        updated = re.sub(r"build check of \d{4}-\d{2}-\d{2}", "build check of %s" % checked.group(1), updated, count=1)
    if updated == text:
        return False
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(updated)
    return True


def build(out_path, _second_pass=False):
    if not check():
        print("\n[ABORT] not building")
        return 1

    entries = manifest()
    data = _write_archive(entries)

    ok, transcript = _verify_extracted(data)
    if not ok:
        print(transcript[-3000:])
        print("\n[ABORT] VERIFY.py failed inside the extracted archive; nothing written")
        return 1

    if _refresh_transcript(transcript) and not _second_pass:
        print("VERIFICATION.md transcript refreshed from this run; rebuilding once")
        return build(out_path, _second_pass=True)

    with open(out_path, "wb") as fh:
        fh.write(data)

    digest = hashlib.sha256(data).hexdigest()
    print("=" * 72)
    print("ARCHIVE BUILT AND VERIFIED")
    print("=" * 72)
    print("path     %s" % out_path)
    print("entries  %d" % len(entries))
    print("size     %s bytes (%.2f MB)" % (format(len(data), ","), len(data) / (1024 * 1024)))
    print("sha256   %s" % digest)
    print("=" * 72)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true",
                        help="Validate the manifest without writing anything")
    parser.add_argument("--list", action="store_true",
                        help="Print the archive manifest (source -> destination)")
    parser.add_argument("--build", metavar="OUT_ZIP",
                        help="Build, verify, and write the archive")
    args = parser.parse_args()

    if args.list:
        for src, dst in manifest():
            print("%-70s -> %s" % (src, dst))
        return 0
    if args.build:
        return build(args.build)
    if args.check:
        ok = check()
        print("\n[OK] the archive manifest is complete and consistent" if ok
              else "\n[FAIL] fix the problems above")
        return 0 if ok else 1
    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
