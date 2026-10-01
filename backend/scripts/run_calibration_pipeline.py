#!/usr/bin/env python3
"""
Run the whole threshold-calibration sequence as one resumable command.

WHY THIS EXISTS

The sequence was four Colab cells. A runtime disconnect part-way through lost
the position, and recovering it meant working out which artefacts already
existed and which commands still had to run. That happened, and the 22-minute
corpus check was repeated for nothing.

Each step records its completion in `.calibration_pipeline_state.json`, keyed by
the image directory and the percentile. On a re-run, a step this run already
completed for the same inputs is SKIPPED, so an interrupted session resumes
where it stopped instead of starting over.

Resume is deliberately NOT based on "does the artefact file exist". Every one of
these artefacts is committed to the repository, so on a fresh clone that test
would skip the work and hand back the previous run's evidence as though it had
just been measured. The state file only ever records work done here.

THE ONE JUDGEMENT THIS DOES NOT MAKE FOR YOU

`--percentile` is required. It is the declared decision - "we admit the
sharpest and largest 99% of the development corpus" - and the thresholds follow
from the measured distribution. The distribution is printed before it is
applied, so the number is visible at the moment it is chosen.

What this will not do is search for a percentile that makes the validation
evidence pass. If ACCEPT cases still fail, that is a finding about the corpus,
and it belongs in Chapter 4 as one.

USAGE

    python backend/scripts/run_calibration_pipeline.py \
        --images-dir aptos2019/train_images --percentile 1.0

    # redo one step and everything that depends on it
    ... --from apply

    # see the plan without running anything
    ... --dry-run
"""

import argparse
import os
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))
CHAPTER4 = os.path.join(REPO_ROOT, "docs", "chapter4")
HANDOVER = os.path.join(REPO_ROOT, "handover2")

PY = sys.executable
STATE = os.path.join(REPO_ROOT, ".calibration_pipeline_state.json")


def _fingerprint(args):
    """Inputs that make a completed step reusable. Change either and the work
    has to be redone, because the artefact would no longer describe them."""
    return {
        "images_dir": os.path.abspath(args.images_dir),
        "percentile": str(args.percentile),
    }


def load_state(args):
    """Steps completed by THIS pipeline for THESE inputs."""
    import json
    if not os.path.exists(STATE):
        return {}
    try:
        with open(STATE, encoding="utf-8") as fh:
            state = json.load(fh)
    except (ValueError, OSError):
        return {}
    if state.get("inputs") != _fingerprint(args):
        # Different corpus or a different declared percentile: nothing carries
        # over, because the artefacts would describe the earlier choice.
        return {}
    return state.get("completed", {})


def record_step(args, key, elapsed):
    import json
    state = {"inputs": _fingerprint(args), "completed": load_state(args)}
    state["completed"][key] = {"seconds": round(elapsed, 1),
                               "at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    with open(STATE, "w", encoding="utf-8") as fh:
        json.dump(state, fh, indent=2)


def artefact(name):
    return os.path.join(CHAPTER4, name)


# Each step: key, human title, the artefact that proves it ran, and the command.
# `slow` marks the two steps measured in tens of minutes, so --skip-slow can
# leave them for a second pass.
def build_steps(args):
    images = args.images_dir
    checkpoint = args.checkpoint or os.path.join(
        REPO_ROOT, "backend", "models", "weights", "efficientnet_b0_dr.pth")

    return [
        {
            "key": "calibrate",
            "title": "Measure all three admission thresholds (train+val only)",
            "proves": artefact("blur_threshold_calibration.json"),
            "cmd": [PY, os.path.join(_HERE, "calibrate_blur_threshold.py"), images],
            "slow": False,
        },
        {
            "key": "apply",
            "title": f"Apply the {args.percentile}th-percentile thresholds",
            # This step edits config.py and validation_module_spec.md rather
            # than creating a file, so its artefact is the spec's derivation
            # record. `--force`/`--from apply` re-runs it.
            "proves": None,
            "cmd": [PY, os.path.join(_HERE, "apply_validation_thresholds.py"),
                    "--percentile", str(args.percentile)],
            "slow": False,
        },
        {
            "key": "evidence",
            "title": "Run the real gates over real images and stated negatives",
            "proves": artefact("validation_test_results.csv"),
            "cmd": [PY, os.path.join(_HERE, "generate_validation_evidence.py"), images],
            "slow": False,
        },
        {
            "key": "downsampling",
            "title": "Decision-preservation check across the full corpus",
            "proves": artefact("gate_downsampling_verification.json"),
            "cmd": [PY, os.path.join(_HERE, "verify_gate_downsampling.py"), images],
            "slow": True,
        },
        {
            "key": "benchmark",
            "title": "End-to-end CPU latency over 30 admitted images",
            "proves": artefact("cpu_end_to_end_benchmark.json"),
            "cmd": [PY, os.path.join(_HERE, "benchmark_cpu_end_to_end.py"),
                    "--images-dir", images, "--checkpoint", checkpoint,
                    "--runs", str(args.runs)],
            "slow": True,
        },
    ]


COLLECT = [
    "blur_threshold_calibration.json",
    "validation_test_results.csv",
    "gate_downsampling_verification.json",
    "cpu_end_to_end_benchmark.json",
    "cpu_end_to_end_benchmark.csv",
    "validation_module_spec.md",
]


def collect(started_at=None):
    """
    Gather the artefacts into handover2/ and zip it.

    `started_at` is the wall-clock time this run began. An artefact older than
    that was not produced here - it is the committed copy - and it is reported
    as STALE rather than shipped silently, so a handover zip cannot quietly
    carry a previous run's evidence.
    """
    import shutil

    os.makedirs(HANDOVER, exist_ok=True)
    present, absent, stale = [], [], []

    def take(src, name, dest_name=None):
        if not os.path.exists(src):
            absent.append(name)
            return
        if started_at is not None and os.path.getmtime(src) < started_at:
            stale.append(name)
            return
        shutil.copy(src, os.path.join(HANDOVER, dest_name or name))
        present.append(name)

    for name in COLLECT:
        take(artefact(name), name)

    take(os.path.join(REPO_ROOT, "backend", "app", "core", "config.py"),
         "config.py")

    archive = shutil.make_archive(
        os.path.join(REPO_ROOT, "calibration_evidence"), "zip", HANDOVER)

    print()
    print("=" * 74)
    print("COLLECTED")
    print("=" * 74)
    for name in present:
        print(f"  [ok]      {name}")
    for name in stale:
        print(f"  [STALE]   {name}  - not written by this run, NOT collected")
    for name in absent:
        print(f"  [MISSING] {name}")
    print()
    print(f"  {len(present)} file(s) -> {os.path.relpath(archive, REPO_ROOT)}")
    if stale:
        print()
        print("  A STALE artefact is the committed copy from an earlier run. It is")
        print("  left out deliberately: shipping it would put evidence in the")
        print("  handover that nobody measured in this session. Re-run the step")
        print("  that produces it with --from <step>.")
    if absent or stale:
        print()
        print("  Re-run this script - steps completed in this session are skipped,")
        print("  so only the outstanding work is repeated.")
    return archive, absent + stale


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images-dir", default="aptos2019/train_images",
                        help="APTOS train_images/ directory")
    parser.add_argument("--percentile", required=True,
                        help="Development-corpus percentile to admit from "
                             "(0.1, 0.5, 1, 2, 5 or 10). This is the declared "
                             "judgement and it is recorded in the spec.")
    parser.add_argument("--checkpoint", default=None,
                        help="Override the checkpoint path for the benchmark")
    parser.add_argument("--runs", type=int, default=30,
                        help="Benchmark requests (default 30)")
    parser.add_argument("--from", dest="start_at", default=None,
                        help="Re-run this step and every step after it, "
                             "ignoring artefacts that already exist")
    parser.add_argument("--skip-slow", action="store_true",
                        help="Leave the corpus check and benchmark for a later pass")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the plan and exit")
    args = parser.parse_args()

    if not os.path.isdir(args.images_dir):
        raise SystemExit(
            f"No image directory at {args.images_dir}.\n"
            "Point --images-dir at the unzipped APTOS train_images/ folder.")

    steps = build_steps(args)
    keys = [s["key"] for s in steps]
    if args.start_at and args.start_at not in keys:
        raise SystemExit(f"--from must be one of {keys}")
    forced_from = keys.index(args.start_at) if args.start_at else None

    print("=" * 74)
    print("THRESHOLD CALIBRATION PIPELINE")
    print("=" * 74)
    print(f"Images     : {args.images_dir}")
    print(f"Percentile : {args.percentile}%  (the declared judgement)")
    print(f"Repository : {REPO_ROOT}")
    print()

    completed = load_state(args)
    if completed:
        print(f"Resuming: {len(completed)} step(s) already completed in this "
              f"session for these inputs ({', '.join(sorted(completed))}).")
        print()

    plan = []
    for index, step in enumerate(steps):
        if args.skip_slow and step["slow"]:
            reason = "skipped (--skip-slow)"
        elif forced_from is not None and index >= forced_from:
            reason = "run (--from)"
        elif step["key"] in completed:
            reason = "SKIP - completed earlier in this session"
        else:
            reason = "run"
        plan.append(reason)
        marker = "  " if reason.startswith("SKIP") or "skipped" in reason else "->"
        print(f" {marker} {step['key']:<13} {step['title']}")
        print(f"      {reason}")
        if step["proves"]:
            print(f"      artefact: {os.path.relpath(step['proves'], REPO_ROOT)}")
    print()

    if args.dry_run:
        print("--dry-run: nothing executed.")
        return 0

    run_started_at = time.time()
    ran, skipped = [], []
    for step, reason in zip(steps, plan):
        if reason.startswith("SKIP") or "skipped" in reason:
            skipped.append(step["key"])
            continue

        print()
        print("-" * 74)
        print(f"[{step['key']}] {step['title']}")
        print("-" * 74)
        started = time.perf_counter()
        result = subprocess.run(step["cmd"], cwd=REPO_ROOT)
        elapsed = time.perf_counter() - started

        if result.returncode != 0:
            print()
            print(f"[STOPPED] Step '{step['key']}' exited {result.returncode} "
                  f"after {elapsed:.0f}s.")
            print("  Nothing after it has run. Fix the cause and re-run this")
            print("  script - the steps that already succeeded will be skipped.")
            collect(run_started_at)
            return result.returncode

        record_step(args, step["key"], elapsed)
        ran.append((step["key"], elapsed))
        print(f"\n[{step['key']}] completed in {elapsed:.0f}s")

    archive, absent = collect(run_started_at)

    print()
    print("=" * 74)
    print(f"Ran {len(ran)} step(s), skipped {len(skipped)}.")
    for key, elapsed in ran:
        print(f"  {key:<13} {elapsed:7.0f}s")
    if skipped:
        print(f"  skipped: {', '.join(skipped)}")
    print("=" * 74)
    print()
    print("Download the archive, then hand it over. In Colab:")
    print("    from google.colab import files")
    print(f"    files.download('{os.path.basename(archive)}')")
    return 1 if absent else 0


if __name__ == "__main__":
    sys.exit(main())
