import io
from fastapi.testclient import TestClient
from PIL import Image

from app import app

client = TestClient(app)


def test_health() -> None:
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_metrics_endpoints() -> None:
    res = client.get("/api/metrics")
    assert res.status_code == 200
    data = res.json()
    assert "ensemble_model" in data
    assert data["ensemble_model"]["accuracy"] > 80.0

    res_roc = client.get("/api/metrics/roc-curve")
    assert res_roc.status_code == 200
    assert res_roc.headers["content-type"] == "image/png"

    res_cm = client.get("/api/metrics/confusion-matrix")
    assert res_cm.status_code == 200
    assert res_cm.headers["content-type"] == "image/png"


def test_rejects_non_image_upload() -> None:
    response = client.post("/api/predict", files={"image": ("bad.txt", b"nope", "text/plain")})
    assert response.status_code == 415


def test_rejects_empty_image() -> None:
    response = client.post("/api/predict", files={"image": ("empty.jpg", b"", "image/jpeg")})
    assert response.status_code == 400


def test_predict_valid_image() -> None:
    # Create a small valid test image in memory
    img = Image.new("RGB", (224, 224), color=(60, 80, 120))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    response = client.post(
        "/api/predict",
        files={"image": ("test.jpg", buf.getvalue(), "image/jpeg")},
    )
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "confidence" in data
    assert "top_predictions" in data
    assert len(data["top_predictions"]) == 3
    assert "model" in data
    assert "platform" in data
    p = data["platform"]
    assert "name" in p
    assert "country" in p
    assert "year_origin" in p
    assert "number_built" in p
    assert "operational_count" in p
    assert "specs" in p


def test_platforms_endpoints() -> None:
    res = client.get("/api/platforms")
    assert res.status_code == 200
    data = res.json()
    assert "total" in data
    assert data["total"] >= 50
    assert len(data["platforms"]) >= 50

    # Specific category filter
    res_air = client.get("/api/platforms?category=aircraft")
    assert res_air.status_code == 200
    air_data = res_air.json()
    assert air_data["total"] >= 20
    for item in air_data["platforms"]:
        assert item["category"] == "aircraft"

    # Specific platform dossier
    res_f22 = client.get("/api/platforms/f-22-raptor")
    assert res_f22.status_code == 200
    f22_data = res_f22.json()
    assert f22_data["name"] == "Lockheed Martin F-22 Raptor"
    assert "United States" in f22_data["country"]
    assert "195" in f22_data["number_built"]


def test_predict_batch() -> None:
    img1 = Image.new("RGB", (224, 224), color=(50, 100, 150))
    buf1 = io.BytesIO()
    img1.save(buf1, format="JPEG")

    img2 = Image.new("RGB", (224, 224), color=(80, 80, 80))
    buf2 = io.BytesIO()
    img2.save(buf2, format="JPEG")

    files = [
        ("images", ("test1.jpg", buf1.getvalue(), "image/jpeg")),
        ("images", ("test2.jpg", buf2.getvalue(), "image/jpeg")),
    ]
    response = client.post("/api/predict/batch", files=files)
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 2
    assert data["successful"] == 2
    assert len(data["results"]) == 2
    assert data["results"][0]["status"] == "success"
    assert data["results"][1]["status"] == "success"


