import sys
sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path
from PIL import Image
from vision import VisionEngine

def test_engine_inference():
    engine = VisionEngine(Path("models"))
    img_path = Path("../ASTRA-Challenge-Starter/challenge-02-vision/images/aircraft/002-GeoFS-F22-Raptor-png.png")
    if not img_path.exists():
        # Fallback to test image
        img = Image.new("RGB", (224, 224), color=(50, 70, 100))
    else:
        img = Image.open(img_path).convert("RGB")

    res = engine.predict(img, "test.png")
    assert "platform" in res
    p = res["platform"]
    print("Detected Category:", res["label"])
    print("Exact Name:", p["name"])
    print("Model/Variant:", p["model_variant"])
    print("Country of Origin:", p["country"])
    print("Year of Origin:", p["year_origin"])
    print("Number Built:", p["number_built"])
    print("Currently Operational:", p["operational_count"])
    print("Role:", p["role"])
    print("Platform Confidence:", p["confidence"])

    assert p["name"]
    assert p["country"]
    assert p["year_origin"] and p["year_origin"] != "Modern"
    assert p["number_built"] and p["number_built"] != "In production"
    assert p["operational_count"]
    assert len(p["runner_ups"]) > 0

if __name__ == "__main__":
    test_engine_inference()
