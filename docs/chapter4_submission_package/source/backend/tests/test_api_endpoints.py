import base64
import pytest
from tests.conftest import create_synthetic_retinal_fundus, image_to_bytes


class TestAuthEndpoints:
    """Verifies authentication endpoints."""

    def test_login_and_me(self, test_client):
        # 1. Login
        login_res = test_client.post(
            "/api/v1/auth/login",
            json={"username": "demo.clinician", "password": "dr_secure_password_2026"},
        )
        assert login_res.status_code == 200
        data = login_res.json()
        assert "access_token" in data
        assert data["user"]["name"] == "Dr. Demo Clinician (Simulated)"

        token = data["access_token"]

        # 2. Get Profile
        me_res = test_client.get(
            "/api/v1/auth/me",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert me_res.status_code == 200
        me_data = me_res.json()
        assert me_data["email"] == "demo.clinician@research-prototype.invalid"
        assert me_data["role"] == "Simulated Reviewer — Research Prototype"


class TestAssessmentEndpoints:
    """Verifies assessment lifecycle and clinical API endpoints."""

    def test_create_and_upload_assessment(self, authed_client):
        test_client = authed_client
        # 1. Create draft assessment
        create_res = test_client.post(
            "/api/v1/assessments",
            json={
                "patientId": "PT-TEST-001",
                "laterality": "OD",
                "cameraModel": "Topcon TRC-NW400",
                "clinicalNotes": "Routine screening check.",
            },
        )
        assert create_res.status_code == 201
        created = create_res.json()
        assessment_id = created["id"]
        assert created["patientId"] == "PT-TEST-001"
        assert created["laterality"] == "OD"
        assert created["status"] == "draft"

        # 2. Upload valid fundus image via multipart
        fundus_img = create_synthetic_retinal_fundus(512, 512, is_retinal=True, blur=False)
        raw_bytes = image_to_bytes(fundus_img, "JPEG")

        upload_res = test_client.post(
            f"/api/v1/assessments/{assessment_id}/upload",
            files={"file": ("test_fundus.jpg", raw_bytes, "image/jpeg")},
        )
        assert upload_res.status_code == 200
        uploaded_data = upload_res.json()
        assert uploaded_data["status"] == "needs_review"  # normalized display status
        assert uploaded_data["modelObservation"] is not None
        assert uploaded_data["modelObservation"]["primaryClassGrade"] == 2
        assert len(uploaded_data["modelObservation"]["classScores"]) == 5
        assert uploaded_data["gradcamUrl"] is not None

        # 3. Retrieve assessment status
        status_res = test_client.get(f"/api/v1/assessments/{assessment_id}/status")
        assert status_res.status_code == 200
        assert status_res.json()["status"] == "result_ready"

        # 4. Retrieve validation details
        val_res = test_client.get(f"/api/v1/assessments/{assessment_id}/validation")
        assert val_res.status_code == 200
        gates = val_res.json()
        assert len(gates) == 3
        assert all(g["status"] == "passed" for g in gates)

        # 5. Retrieve result details
        result_res = test_client.get(f"/api/v1/assessments/{assessment_id}/result")
        assert result_res.status_code == 200
        obs = result_res.json()
        assert obs["primaryClassLabel"] == "Moderate NPDR"
        assert 0.0 <= obs["primaryScore"] <= 1.0

        # 6. Retrieve audit trail
        audit_res = test_client.get(f"/api/v1/assessments/{assessment_id}/audit")
        assert audit_res.status_code == 200
        audit_events = audit_res.json()
        assert len(audit_events) >= 3

        # 7. Submit Professional Review (Clinician signs off)
        review_res = test_client.post(
            f"/api/v1/assessments/{assessment_id}/review",
            json={
                "agreement": "agree",
                "reviewerAssessedGrade": 2,
                "reviewerAssessedGradeLabel": "Grade 2: Moderate NPDR",
                "justificationNotes": "Confirmed presence of parafoveal microaneurysms.",
            },
        )
        assert review_res.status_code == 200
        reviewed = review_res.json()
        assert reviewed["status"] == "completed"
        assert reviewed["clinicianReview"] is not None
        assert reviewed["clinicianReview"]["signatureHash"].startswith("SIG-SHA256-")

        # 8. Download PDF Report
        report_res = test_client.get(f"/api/v1/assessments/{assessment_id}/report")
        assert report_res.status_code == 200
        assert report_res.headers["content-type"] == "application/pdf"
        assert report_res.content.startswith(b"%PDF-")

        # Alias download route
        alias_res = test_client.get(f"/api/v1/reports/{assessment_id}/pdf")
        assert alias_res.status_code == 200
        assert alias_res.content.startswith(b"%PDF-")

    def test_direct_data_url_creation_with_gate_failure(self, authed_client):
        test_client = authed_client
        """Verify POST /assessments with simulated Gate 2 rejection."""
        fundus_img = create_synthetic_retinal_fundus(512, 512, is_retinal=False)
        raw_bytes = image_to_bytes(fundus_img, "JPEG")
        data_url = f"data:image/jpeg;base64,{base64.b64encode(raw_bytes).decode('utf-8')}"

        res = test_client.post(
            "/api/v1/assessments",
            json={
                "patientId": "PT-TEST-REJECT-02",
                "laterality": "OS",
                "imageDataUrl": data_url,
                "simulateGateFailure": 2,
            },
        )
        assert res.status_code == 201
        data = res.json()
        assert data["status"] == "rejected"
        assert data["modelObservation"] is None
        # Gate 2 must be failed, Gate 3 pending
        assert data["validationGates"][1]["status"] == "failed"
        assert data["validationGates"][2]["status"] == "pending"

    def test_review_friction_justification_rule(self, authed_client):
        test_client = authed_client
        """Clinical Governance Invariant: Overriding AI requires >= 15 char justification."""
        # Create and run valid assessment
        fundus_img = create_synthetic_retinal_fundus(512, 512, is_retinal=True)
        raw_bytes = image_to_bytes(fundus_img, "JPEG")
        data_url = f"data:image/jpeg;base64,{base64.b64encode(raw_bytes).decode('utf-8')}"

        res = test_client.post(
            "/api/v1/assessments",
            json={
                "patientId": "PT-OVERRIDE-TEST",
                "laterality": "OD",
                "imageDataUrl": data_url,
            },
        )
        assessment_id = res.json()["id"]

        # Attempt to disagree with insufficient explanation (<15 chars)
        short_res = test_client.post(
            f"/api/v1/assessments/{assessment_id}/review",
            json={
                "agreement": "disagree",
                "reviewerAssessedGrade": 0,
                "justificationNotes": "Too short",  # 9 chars < 15
            },
        )
        assert short_res.status_code == 422
        assert "at least 15 characters" in short_res.json()["detail"]

        # Valid override with sufficient explanation (>= 15 chars)
        valid_res = test_client.post(
            f"/api/v1/assessments/{assessment_id}/review",
            json={
                "agreement": "disagree",
                "reviewerAssessedGrade": 1,
                "justificationNotes": "Microaneurysms only. No blot hemorrhages seen in 4 quadrants.",
            },
        )
        assert valid_res.status_code == 200
        assert valid_res.json()["status"] == "completed"


