"""
The documented validation thresholds must match the ones the code enforces.

An external reviewer found the Chapter Four documents quoting a Laplacian
sharpness threshold of both 60 and 100, and an aperture aspect-ratio window of
0.65-1.65, 0.75-1.33 and 0.80-1.25, across different files. A specification that
contradicts itself cannot be verified against an implementation, and the
implementation silently wins.

These tests read the committed documents and assert that no contradictory value
appears. They are deliberately narrow: they check the small set of numeric
thresholds a reviewer would cross-reference, not prose.
"""

import os
import re

import pytest

from app.core.config import settings

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DOC_DIRS = [os.path.join(REPO_ROOT, "docs")]


# Documents whose job is to record what was wrong must be allowed to quote it.
# Excluding them by name keeps the rule strict everywhere else; a blanket
# "ignore any line mentioning a fix" would let real regressions slip through.
CORRECTION_RECORDS = {
    "REVIEWER_RESPONSE.md",
    "independent_thesis_qa_gate_audit.md",
    "screenshot_evidence_manifest.md",
}


def iter_markdown(include_correction_records=True):
    for root_dir in DOC_DIRS:
        for dirpath, _dirnames, filenames in os.walk(root_dir):
            for name in filenames:
                if not name.endswith(".md"):
                    continue
                if not include_correction_records and name in CORRECTION_RECORDS:
                    continue
                path = os.path.join(dirpath, name)
                with open(path, "r", encoding="utf-8", errors="replace") as fh:
                    yield os.path.relpath(path, REPO_ROOT), fh.read()


def test_aspect_ratio_window_is_quoted_consistently():
    """
    Gate 2 accepts 0.65 <= aspect ratio <= 1.65. Any other window quoted beside
    the words "aspect ratio" is a documentation defect.
    """
    allowed = {("0.65", "1.65")}
    offenders = []

    for rel, text in iter_markdown(include_correction_records=False):
        for match in re.finditer(r"[Aa]spect [Rr]atio[^\n]{0,80}", text):
            fragment = match.group(0)
            for lo, hi in re.findall(r"(\d\.\d{1,2})\s*(?:-|–|to|\\leq[^\d]*)\s*(\d\.\d{1,2})", fragment):
                if (lo.rstrip("0").rstrip("."), hi.rstrip("0").rstrip(".")) != ("0.65", "1.65"):
                    if (lo, hi) not in allowed:
                        offenders.append(f"{rel}: {fragment.strip()}")

    assert not offenders, (
        "Documented aspect-ratio window disagrees with gate2_relevance.py "
        "(0.65-1.65):\n  " + "\n  ".join(offenders)
    )


def test_laplacian_threshold_is_quoted_consistently():
    """
    Gate 3 rejects below LAPLACIAN_BLUR_THRESHOLD. No document may quote a
    different sharpness threshold.
    """
    expected = settings.LAPLACIAN_BLUR_THRESHOLD  # 60.0
    offenders = []

    for rel, text in iter_markdown(include_correction_records=False):
        for match in re.finditer(r"[Ll]aplacian[^\n]{0,60}", text):
            fragment = match.group(0)
            # Only the FIRST threshold after the word, so that neighbouring
            # illumination and contrast thresholds are not misattributed to it.
            found = re.search(
                r"(?:[Tt]hreshold|>=|\\ge|≥)\s*\$?\\?[a-z]*\s*(\d{2,3}(?:\.\d)?)", fragment
            )
            if found and abs(float(found.group(1)) - expected) > 1e-6:
                offenders.append(f"{rel}: {fragment.strip()}")

    assert not offenders, (
        f"Documented Laplacian threshold disagrees with "
        f"LAPLACIAN_BLUR_THRESHOLD={expected}:\n  " + "\n  ".join(offenders)
    )


@pytest.mark.parametrize(
    "setting_name,expected",
    [("RETINAL_RED_RATIO_MIN", 1.15), ("RETINAL_MIN_COVERAGE", 0.20),
     ("RETINAL_MAX_COVERAGE", 0.98), ("CONTRAST_THRESHOLD", 18.0),
     ("ILLUMINATION_EXTREME_RATIO_MAX", 0.35), ("LAPLACIAN_BLUR_THRESHOLD", 60.0)],
)
def test_config_values_are_what_the_documents_describe(setting_name, expected):
    """
    Pins the thresholds themselves. If one is deliberately changed, this fails
    and forces the documents to be revisited in the same commit.
    """
    assert getattr(settings, setting_name) == expected


def test_no_document_claims_a_fabricated_clinician_identity():
    """
    The system must not present a fabricated professional registration number
    or an affiliation with a real institution, in code, documents or figures.
    """
    banned = ["GMC-7492104", "Adaeze Okonjo", "St. Jude Retinal",
              "a.okonjo@retina-clinic.nhs.uk"]
    offenders = []

    for rel, text in iter_markdown(include_correction_records=False):
        for term in banned:
            if term in text:
                offenders.append(f"{rel}: contains {term!r}")

    assert not offenders, "Fabricated clinical identity in documentation:\n  " + "\n  ".join(offenders)
