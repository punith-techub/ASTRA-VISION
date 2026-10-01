import pytest
from PIL import Image
import multi_object_engine as moe


def test_compute_box_iou():
    boxA = (10, 10, 50, 50)
    boxB = (10, 10, 50, 50)
    assert moe.compute_box_iou(boxA, boxB) == 1.0

    boxC = (100, 100, 200, 200)
    assert moe.compute_box_iou(boxA, boxC) == 0.0


def test_multi_target_processing():
    # Create test synthetic image
    img = Image.new("RGB", (600, 400), color=(100, 150, 255))

    mock_gemini = {
        "target_count": 2,
        "detected_targets": [
            {
                "id": 1,
                "box_2d": [200, 200, 400, 400],
                "exact_name": "Tactical Reconnaissance Unit Alpha",
                "designation": "TRU-01",
                "confidence": 99.5,
                "category": "aircraft",
            },
            {
                "id": 2,
                "box_2d": [200, 450, 400, 650],
                "exact_name": "Tactical Reconnaissance Unit Beta",
                "designation": "TRU-02",
                "confidence": 99.5,
                "category": "aircraft",
            }
        ]
    }

    res = moe.process_multi_target_recon(img, mock_gemini)
    assert res["target_count"] >= 2
    assert len(res["detected_targets"]) >= 2
    assert "data:image/jpeg;base64," in res["annotated_image_url"]
    t1 = res["detected_targets"][0]
    assert t1["exact_name"] == "Tactical Reconnaissance Unit Alpha"
    assert t1["id"] == 1
    assert "data:image/" in t1["crop_url"]
    assert "year_origin" in t1 and t1["year_origin"] != "Modern"
    assert "number_built" in t1 and t1["number_built"] != "In production"
