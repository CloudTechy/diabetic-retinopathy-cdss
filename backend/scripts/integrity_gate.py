#!/usr/bin/env python3
"""
Run the evidence integrity gate locally, the same checks CI enforces.

Use this before committing anything that touches the model, the validation
gates, the evidence artefacts, the generated report, or the Chapter Four
documents. It is fast and needs no dataset, no GPU and no PyTorch.

    python backend/scripts/integrity_gate.py

Exit code 0 means the change is safe to commit. Anything else names what broke
and why it matters.

To install it as a pre-commit hook so it cannot be forgotten:

    python backend/scripts/integrity_gate.py --install-hook
"""

import argparse
import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BACKEND = os.path.join(REPO_ROOT, "backend")

HOOK = r"""#!/bin/sh
# Evidence integrity gate - installed by backend/scripts/integrity_gate.py
# Bypass with --no-verify only if you know why the gate is wrong.
python backend/scripts/integrity_gate.py || {
    echo ""
    echo "Commit blocked by the evidence integrity gate."
    echo "Run: python backend/scripts/integrity_gate.py"
    exit 1
}
"""


def _python():
    """Prefer the backend virtualenv, which has the test dependencies."""
    for candidate in (
        os.path.join(BACKEND, ".venv", "Scripts", "python.exe"),
        os.path.join(BACKEND, ".venv", "bin", "python"),
    ):
        if os.path.exists(candidate):
            return candidate
    return sys.executable


def run(label, argv, cwd):
    print(f"\n>>> {label}")
    result = subprocess.run(argv, cwd=cwd)
    ok = result.returncode == 0
    print(f"    {'PASS' if ok else 'FAIL'}  {label}")
    return ok


def install_hook():
    hooks = os.path.join(REPO_ROOT, ".git", "hooks")
    if not os.path.isdir(hooks):
        print(f"No .git/hooks directory at {hooks}")
        return 1
    path = os.path.join(hooks, "pre-commit")
    with open(path, "w", newline="\n", encoding="utf-8") as fh:
        fh.write(HOOK)
    try:
        os.chmod(path, 0o755)
    except OSError:
        pass
    print(f"Installed pre-commit hook at {path}")
    print("Every commit now runs the gate. Bypass with 'git commit --no-verify'.")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--install-hook", action="store_true",
                        help="Install as a git pre-commit hook")
    parser.add_argument("--full", action="store_true",
                        help="Also run the complete backend test suite")
    args = parser.parse_args()

    if args.install_hook:
        return install_hook()

    py = _python()
    env_note = "backend/.venv" if py != sys.executable else "system python"
    print("=" * 74)
    print("EVIDENCE INTEGRITY GATE")
    print("=" * 74)
    print(f"Interpreter: {env_note}")

    checks = [
        ("Integrity gate + specification consistency",
         [py, "-m", "pytest", "tests/test_editor_integrity_gate.py",
          "tests/test_spec_doc_consistency.py", "-q"], BACKEND),
        ("Reported metrics recompute from raw predictions",
         [py, os.path.join("backend", "scripts", "analyze_clinical_metrics.py")], REPO_ROOT),
        ("Submission package matches its source artefacts",
         [py, os.path.join("backend", "scripts", "assemble_submission_package.py"),
          "--check"], REPO_ROOT),
    ]
    if args.full:
        checks.append(("Full backend suite",
                       [py, "-m", "pytest", "tests/", "-q"], BACKEND))

    results = [run(label, argv, cwd) for label, argv, cwd in checks]

    print("\n" + "=" * 74)
    if all(results):
        print("GATE PASSED - safe to commit.")
        return 0

    failed = [label for (label, _, _), ok in zip(checks, results) if not ok]
    print("GATE FAILED")
    for label in failed:
        print(f"  - {label}")
    print()
    print("These rules encode the Chapter Four QA review's requirements. A failure")
    print("means a change would reintroduce something that review rejected. Fix the")
    print("cause; if a rule is genuinely wrong, widen it explicitly and say why in")
    print("the same commit rather than deleting it.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
