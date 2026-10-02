import os
import pytest
from fastapi.testclient import TestClient
from tests.conftest import create_synthetic_retinal_fundus, image_to_bytes
import io
import base64

def create_mock_fundus_image():
    img = create_synthetic_retinal_fundus(width=800, height=800)
    b = image_to_bytes(img)
    return base64.b64encode(b).decode('utf-8')

def test_real_integration_with_real_model(authed_client):
    from app.core.config import settings
    import app.services.ai_service as ai_service
    
    old_engine = os.environ.get("AI_INFERENCE_ENGINE", "mock")
    os.environ["AI_INFERENCE_ENGINE"] = "real"
    settings.MODEL_CHECKPOINT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../models/weights/efficientnet_b0_dr.pth'))
    
    # Store old singleton just in case
    old_singleton = ai_service._service_singleton
    ai_service._service_singleton = None
    
    try:
        client = authed_client
    
        b64_img = create_mock_fundus_image()
        payload = {
            "patientId": "REAL_TEST_PATIENT",
            "laterality": "OS",
            "cameraModel": "Test Camera",
            "isMydriatic": False,
            "imageDataUrl": f"data:image/jpeg;base64,{b64_img}"
        }
    
        # Set up auth headers
        headers = {}
    
        # Post assessment
        response = client.post("/api/v1/assessments", json=payload, headers=headers)
        assert response.status_code in (200, 201), f"Failed: {response.text}"
        
        data = response.json()
        assert data["status"] in ("result_ready", "needs_review")
        
        # Verify inference output matches expected structure
        assessment_id = data["id"]
        
        response = client.get(f"/api/v1/assessments/{assessment_id}", headers=headers)
        assert response.status_code == 200
        
        data = response.json()
        ai_result = data.get("modelObservation", data.get("aiResult"))
        assert ai_result is not None
        
        # Check probabilities
        assert len(ai_result["classScores"]) == 5
        predicted_class = ai_result["primaryClassGrade"]
        
        probs = [c["score"] for c in ai_result["classScores"]]
        max_prob = max(probs)
        max_idx = probs.index(max_prob)
        
        assert predicted_class == max_idx
        
        # Check grad-cam
        gradcam_url = data.get("gradcamUrl")
        assert gradcam_url is not None
    finally:
        os.environ["AI_INFERENCE_ENGINE"] = old_engine
        ai_service._service_singleton = old_singleton


