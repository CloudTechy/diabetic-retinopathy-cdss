"""
Calibrated outputs are labelled as class probabilities; anything without a
verified calibration record keeps the model-generated class score wording.

The report is built through the API twice: once with the real engine class
carrying the committed temperature-scaling artifact (a stub network with fixed
logits, so the vector is known), once with the simulated engine, which records
no calibration. Every Paragraph and table the report asks ReportLab to draw is
recorded, so the test reads the labels the PDF prints and checks that each
wrapped label fits inside its column.
"""
import json
import math
import os

import numpy as np
import pytest
from reportlab.platypus import Paragraph

import app.services.report_service as report_service
from app.services.score_terminology import (
    CALIBRATION_EXPLANATION,
    UNCALIBRATED_EXPLANATION,
    is_probability_vector,
    recorded_temperature,
    score_terms,
)
from tests.conftest import create_synthetic_retinal_fundus, image_to_bytes
from tests.test_probability_calibration import STUB_LOGITS, _softmax, _stub_engine, requires_torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ARTIFACT = os.path.join(ROOT, "docs", "chapter4", "probability_calibration", "temperature_scaling_artifact.json")
CALIBRATED_VERSION = "EfficientNet-B0-DR-v1 (fixed weights; temperature-scaled T=1.093921, ts-v2)"
OBSOLETE = ("Relative Score", "Score Breakdown", "feature activation", "algorithmic association")
PADDING = 12  # ReportLab's default left + right cell padding, in points

# The calibrated vector the API served for REC-2026-6954 (held-out image c6c2bad91f23).
SERVED = [0.000693969914342396, 0.007727745597431056, 0.8843014501550295, 0.05967577356239384, 0.04760106077080314]


def _scores(values):
    return [{"grade": g, "label": f"Grade {g}", "score": v} for g, v in enumerate(values)]


# ---------------------------------------------------------------------------
# the rule
# ---------------------------------------------------------------------------

def test_recorded_calibration_with_a_valid_vector_is_a_probability():
    terms = score_terms(CALIBRATED_VERSION, _scores(SERVED))
    assert terms.calibrated and terms.temperature == pytest.approx(1.093921)
    assert terms.highest == "Highest Class Probability"
    assert terms.per_class == "Temperature-Scaled Class Probability"
    assert terms.distribution == "Five-Class Probability Distribution"
    assert terms.explanation == CALIBRATION_EXPLANATION


@pytest.mark.parametrize("version", [None, "", "EfficientNet-B0-DR-v1 (fixed weights)", "Simulated engine"])
def test_no_recorded_calibration_stays_a_model_generated_score(version):
    terms = score_terms(version, _scores(SERVED))
    assert not terms.calibrated and terms.temperature is None
    assert terms.per_class == "Model-Generated Class Score"
    assert terms.explanation == UNCALIBRATED_EXPLANATION
    assert "probabilit" not in (terms.highest + terms.per_class + terms.distribution).lower()


@pytest.mark.parametrize("values", [
    [v * 2 for v in SERVED],               # does not sum to 1
    [1.2, -0.2, 0.0, 0.0, 0.0],            # outside 0-1
    [float("nan"), 0.5, 0.5, 0.0, 0.0],    # not finite
    SERVED[:4],                            # a class missing
])
def test_a_malformed_vector_is_never_called_a_probability(values):
    assert not score_terms(CALIBRATED_VERSION, _scores(values)).calibrated


def test_served_vector_is_finite_in_range_and_sums_to_one():
    assert is_probability_vector(SERVED)
    assert all(math.isfinite(v) and 0 <= v <= 1 for v in SERVED)
    assert abs(sum(SERVED) - 1) < 1e-9
    assert recorded_temperature(CALIBRATED_VERSION) == pytest.approx(1.093921)


def test_no_label_uses_obsolete_wording():
    for terms in (score_terms(CALIBRATED_VERSION, _scores(SERVED)), score_terms(None, None)):
        text = " ".join([terms.highest, terms.per_class, terms.distribution, terms.explanation])
        for phrase in OBSOLETE:
            assert phrase.lower() not in text.lower()


# ---------------------------------------------------------------------------
# the report
# ---------------------------------------------------------------------------

def _report(client, monkeypatch):
    paragraphs, tables = [], []
    real_paragraph, real_table = report_service.Paragraph, report_service.Table

    class RecordingParagraph(real_paragraph):
        def __init__(self, text, *args, **kwargs):
            super().__init__(text, *args, **kwargs)
            paragraphs.append(self)

    class RecordingTable(real_table):
        def __init__(self, data, colWidths=None, *args, **kwargs):
            tables.append((data, colWidths))
            super().__init__(data, colWidths, *args, **kwargs)

    monkeypatch.setattr(report_service, "Paragraph", RecordingParagraph)
    monkeypatch.setattr(report_service, "Table", RecordingTable)

    created = client.post("/api/v1/assessments", json={
        "patientId": "PT-TERMS-001", "laterality": "OD", "cameraModel": "Topcon TRC-NW400",
        "clinicalNotes": "terminology test"})
    assert created.status_code == 201
    aid = created.json()["id"]
    up = client.post(f"/api/v1/assessments/{aid}/upload", files={
        "file": ("f.jpg", image_to_bytes(create_synthetic_retinal_fundus(512, 512), "JPEG"), "image/jpeg")})
    assert up.status_code == 200, up.text
    obs = up.json()["modelObservation"]
    grade = obs["primaryClassGrade"]
    rev = client.post(f"/api/v1/assessments/{aid}/review", json={
        "agreement": "agree", "reviewerAssessedGrade": grade,
        "reviewerAssessedGradeLabel": f"Grade {grade}", "justificationNotes": "Terminology test review."})
    assert rev.status_code == 200, rev.text
    pdf = client.get(f"/api/v1/assessments/{aid}/report")
    assert pdf.status_code == 200 and pdf.content.startswith(b"%PDF-")
    return obs, [p.text for p in paragraphs], tables


def _score_table(tables):
    for data, widths in tables:
        header = data[0]
        if header and isinstance(header[0], Paragraph) and "ICDR Stage" in header[0].text:
            return data, widths
    raise AssertionError("the report drew no five-class table")


def _model_tables(tables):
    """The two tables of section 2 that carry the score labels: the observation table and the five-class table."""
    def texts(data):
        return " ".join(cell.text if isinstance(cell, Paragraph) else str(cell) for row in data for cell in row)
    picked = [(d, w) for d, w in tables if "Primary Candidate Class:" in texts(d) or "ICDR Stage" in texts(d)]
    assert len(picked) == 2, len(picked)
    return picked


def _assert_wrapped_cells_fit(tables):
    """A Paragraph cell wraps at spaces; its longest word must still fit the column, or it is clipped."""
    for data, widths in _model_tables(tables):
        for row in data:
            for c, cell in enumerate(row):
                if isinstance(cell, Paragraph):
                    assert cell.minWidth() + PADDING <= widths[c], (cell.text, cell.minWidth(), widths[c])


@requires_torch
def test_calibrated_report_prints_probability_labels(authed_client, monkeypatch):
    import app.services.ai_service as ai_svc
    with open(ARTIFACT, encoding="utf-8") as fh:
        artifact = json.load(fh)
    old = ai_svc._service_singleton
    ai_svc._service_singleton = _stub_engine(artifact)
    try:
        obs, texts, tables = _report(authed_client, monkeypatch)
    finally:
        ai_svc._service_singleton = old

    # the API vector is the calibrated one; the grade is its argmax
    expected = _softmax(STUB_LOGITS, float(artifact["temperature"]))
    served = [c["score"] for c in sorted(obs["classScores"], key=lambda c: c["grade"])]
    assert np.max(np.abs(np.array(served) - expected)) < 1e-6
    assert obs["primaryClassGrade"] == int(np.argmax(expected))

    joined = "\n".join(texts)
    assert "Highest Class Probability:" in joined
    assert "Five-Class Probability Distribution" in joined
    assert CALIBRATION_EXPLANATION in joined
    assert "Temperature-scaled (T = 1.0939, validation partition)" in joined
    for phrase in OBSOLETE:
        assert phrase.lower() not in joined.lower(), phrase

    data, widths = _score_table(tables)
    assert data[0][2].text == "Temperature-Scaled Class Probability"
    # each printed value is the served value rounded to two decimals, in grade order
    assert [row[2] for row in data[1:]] == [f"{v:.2f}" for v in served]
    _assert_wrapped_cells_fit(tables)


def test_uncalibrated_report_keeps_model_generated_score_labels(authed_client, monkeypatch):
    obs, texts, tables = _report(authed_client, monkeypatch)   # simulated engine: no calibration record
    assert obs.get("calibrationTemperature") is None
    joined = "\n".join(texts)
    assert "Highest Model-Generated Class Score:" in joined
    assert UNCALIBRATED_EXPLANATION in joined
    assert "Highest Class Probability" not in joined
    assert "Temperature-Scaled Class Probability" not in joined
    data, _ = _score_table(tables)
    assert data[0][2].text == "Model-Generated Class Score"
    _assert_wrapped_cells_fit(tables)
    # Grad-CAM labels stay Grad-CAM labels
    assert "Target Saliency Layer:" in joined
