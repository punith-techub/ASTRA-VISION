from pathlib import Path
import io
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app import app
from vision import VisionEngine

client = TestClient(app)


def test_learning_endpoint() -> None:
    response = client.get("/api/learning")
    assert response.status_code == 200
    data = response.json()
    assert "health_score" in data
    assert "total_evaluations" in data
    assert "accuracy_rate" in data


def test_v5_two_stage_prediction_and_learning() -> None:
    # Generate test image
    img = Image.new("RGB", (224, 224), color=(80, 100, 140))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    # 1st Scan: Local model attempts first, API checks, verification occurs
    response = client.post(
        "/api/predict",
        files={"image": ("test_jet.jpg", buf.getvalue(), "image/jpeg")},
    )
    assert response.status_code == 200
    data = response.json()
    assert "verification" in data
    v = data["verification"]
    assert "status" in v
    assert "local_answer" in v
    assert "verified_answer" in v
    assert "learned_samples" in v

    # 2nd Scan with same/similar image: Model queries memory first and benefits from learned exemplar!
    buf.seek(0)
    response_2 = client.post(
        "/api/predict",
        files={"image": ("test_jet_repeat.jpg", buf.getvalue(), "image/jpeg")},
    )
    assert response_2.status_code == 200
    data_2 = response_2.json()
    assert "verification" in data_2
    assert data_2["platform"]["year_origin"] != "Modern"
    assert data_2["platform"]["number_built"] != "In production"
    if "detected_targets" in data_2 and len(data_2["detected_targets"]) > 0:
        t0 = data_2["detected_targets"][0]
        assert t0.get("year_origin") and t0["year_origin"] != "Modern"
        assert t0.get("number_built") and t0["number_built"] != "In production"


def test_offline_prediction_skips_api_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that when internet is off, API keys are skipped and local trained model result is returned immediately."""
    import omni_intelligence

    # Simulate internet turned off
    monkeypatch.setattr(omni_intelligence, "is_internet_available", lambda *args, **kwargs: False)

    img = Image.new("RGB", (224, 224), color=(30, 45, 75))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    response = client.post(
        "/api/predict",
        files={"image": ("offline_test.jpg", buf.getvalue(), "image/jpeg")},
    )
    assert response.status_code == 200
    data = response.json()

    # Must give result from trained local model
    assert data.get("is_offline") is True
    assert "prediction" in data
    assert "label" in data
    assert "confidence" in data
    assert isinstance(data["confidence"], (int, float))
    assert 0.0 <= data["confidence"] <= 100.0

    # Verification must state offline
    assert "verification" in data
    assert data["verification"].get("is_offline") is True
    assert data["verification"].get("status") == "offline"
    assert data["verification"].get("verified_answer") is None

    # Platform must have actual confidence score
    assert "platform" in data
    assert "name" in data["platform"]
    assert "confidence" in data["platform"]
    assert isinstance(data["platform"]["confidence"], (int, float))
