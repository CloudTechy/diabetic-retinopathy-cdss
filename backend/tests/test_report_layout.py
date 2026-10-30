"""
The PDF report's tables must not draw text over a neighbouring cell.

ReportLab draws a plain-string table cell on one line at its full width; it
never wraps it. "Gate 2: Technical retinal-image relevance" was wider than its
column and ran into the PASSED cell beside it, and the bold "Model-Generated
Class Score" header ran past the edge of its column. This generates a real
report through the API, records every table the report asks ReportLab to draw,
and checks that each plain-string cell fits inside its column. Paragraph cells
wrap within their column and are exempt.
"""
from reportlab.lib.units import inch
from reportlab.pdfbase.pdfmetrics import stringWidth

import app.services.report_service as report_service
from tests.conftest import create_synthetic_retinal_fundus, image_to_bytes

PADDING = 12  # ReportLab's default left + right cell padding, in points


def _completed_assessment(client):
    created = client.post("/api/v1/assessments", json={
        "patientId": "PT-LAYOUT-001", "laterality": "OD", "cameraModel": "Topcon TRC-NW400",
        "clinicalNotes": "report layout test"})
    assert created.status_code == 201
    aid = created.json()["id"]
    up = client.post(f"/api/v1/assessments/{aid}/upload",
                     files={"file": ("f.jpg", image_to_bytes(create_synthetic_retinal_fundus(512, 512), "JPEG"), "image/jpeg")})
    assert up.status_code == 200, up.text
    rev = client.post(f"/api/v1/assessments/{aid}/review", json={
        "agreement": "agree", "reviewerAssessedGrade": 2, "reviewerAssessedGradeLabel": "Grade 2: Moderate NPDR",
        "justificationNotes": "Layout test review."})
    assert rev.status_code == 200, rev.text
    return aid


def test_report_table_cells_fit_their_columns(authed_client, monkeypatch):
    drawn = []
    real_table = report_service.Table

    class RecordingTable(real_table):
        def __init__(self, data, colWidths=None, *args, **kwargs):
            drawn.append((data, colWidths))
            super().__init__(data, colWidths, *args, **kwargs)

    monkeypatch.setattr(report_service, "Table", RecordingTable)
    aid = _completed_assessment(authed_client)
    report = authed_client.get(f"/api/v1/assessments/{aid}/report")
    assert report.status_code == 200 and report.content.startswith(b"%PDF-")
    assert drawn, "the report drew no tables"

    overflow = []
    for data, widths in drawn:
        if not widths:
            continue
        for r, row in enumerate(data):
            for c, cell in enumerate(row):
                if not isinstance(cell, str) or not cell:
                    continue
                # Measured in the widest face the report uses for table text, so a
                # header in bold cannot pass on the strength of the regular face.
                need = stringWidth(cell, "Helvetica-Bold", 8) + PADDING
                if need > widths[c]:
                    overflow.append("%r needs %.2f in, column is %.2f in" % (cell, need / inch, widths[c] / inch))
    assert not overflow, "table text wider than its column:\n  " + "\n  ".join(overflow)
