from __future__ import annotations

import io
import json
import os
from pathlib import Path
from typing import Annotated

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile

load_dotenv()
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

import defense_knowledge as dk
from vision import VisionEngine

ROOT = Path(__file__).resolve().parent
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
SUPPORTED_TYPES = {"image/jpeg", "image/png", "image/webp"}

app = FastAPI(title="ASTRA VISION", version="5.0.0")
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")
engine = VisionEngine(ROOT / "models")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(ROOT / "static" / "index.html")


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", **engine.status()}


@app.get("/api/learning")
def learning() -> dict:
    return engine.online_trainer.status()


@app.get("/api/platforms")
def list_platforms(category: str | None = None) -> dict:
    """Returns catalog of indexed military platforms with specifications and stats."""
    if category:
        platforms = dk.get_platforms_by_category(category)
    else:
        platforms = dk.all_platforms()
    return {
        "total": len(platforms),
        "platforms": [
            {
                "id": p.get("id", ""),
                "name": p.get("name", ""),
                "designation": p.get("designation", ""),
                "category": p.get("category", "military-platform"),
                "country": p.get("country", ""),
                "year_origin": p.get("year_origin", ""),
                "operational_count": p.get("operational_count", ""),
                "role": p.get("role", ""),
            }
            for p in platforms
        ],
    }


@app.get("/api/platforms/{platform_id}")
def get_platform_dossier(platform_id: str) -> dict:
    """Returns complete intelligence dossier for a specific military platform."""
    p = dk.get_platform(platform_id)
    if not p:
        raise HTTPException(404, f"Platform '{platform_id}' not found in defense knowledge base.")
    return p


@app.get("/api/metrics")
def metrics() -> dict:
    metrics_path = ROOT / "models" / "metrics.json"
    if not metrics_path.exists():
        raise HTTPException(404, "Metrics not found. Run train.py to generate them.")
    return json.loads(metrics_path.read_text(encoding="utf-8"))


@app.get("/api/metrics/roc-curve")
def roc_curve() -> FileResponse:
    roc_path = ROOT / "models" / "roc_curve.png"
    if not roc_path.exists():
        raise HTTPException(404, "ROC curve plot not found.")
    return FileResponse(roc_path, media_type="image/png")


@app.get("/api/metrics/confusion-matrix")
def confusion_matrix() -> FileResponse:
    cm_path = ROOT / "models" / "confusion_matrix.png"
    if not cm_path.exists():
        raise HTTPException(404, "Confusion matrix plot not found.")
    return FileResponse(cm_path, media_type="image/png")


@app.post("/api/predict")
async def predict(
    image: Annotated[UploadFile, File(...)],
    x_gemini_key: Annotated[str | None, Header(alias="X-Gemini-Key")] = None,
    api_key: Annotated[str | None, Form()] = None,
) -> dict:
    if image.content_type not in SUPPORTED_TYPES:
        raise HTTPException(415, "Upload a JPG, PNG, or WebP image.")

    raw = await image.read(MAX_UPLOAD_BYTES + 1)
    if not raw:
        raise HTTPException(400, "The uploaded image is empty.")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Image is larger than the 10 MB limit.")

    try:
        with Image.open(io.BytesIO(raw)) as opened:
            opened.verify()
        with Image.open(io.BytesIO(raw)) as opened:
            picture = opened.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(400, "This file is not a readable image.")

    recon_key = x_gemini_key or api_key
    return engine.predict(picture, image.filename or "upload", api_key=recon_key)


@app.post("/api/predict/batch")
async def predict_batch(
    images: list[UploadFile] = File(...),
) -> dict:
    """Processes multiple images in a single batch request for high-throughput evaluation."""
    if not images:
        raise HTTPException(400, "No images provided for batch processing.")
    if len(images) > 30:
        raise HTTPException(400, "Batch processing is limited to 30 images per request.")

    results = []
    for img in images:
        filename = img.filename or "upload"
        if img.content_type not in SUPPORTED_TYPES:
            results.append({"filename": filename, "error": "Unsupported image format", "status": "failed"})
            continue

        raw = await img.read(MAX_UPLOAD_BYTES + 1)
        if not raw or len(raw) > MAX_UPLOAD_BYTES:
            results.append({"filename": filename, "error": "Invalid or oversized image", "status": "failed"})
            continue

        try:
            with Image.open(io.BytesIO(raw)) as opened:
                opened.verify()
            with Image.open(io.BytesIO(raw)) as opened:
                picture = opened.convert("RGB")

            # Local prediction for high-speed batch processing
            pred = engine.predict_local(picture, filename)
            pred.pop("img_feats", None)

            results.append({
                "filename": filename,
                "prediction": pred.get("prediction"),
                "label": pred.get("label"),
                "confidence": pred.get("confidence"),
                "platform": pred.get("platform", {}).get("name"),
                "designation": pred.get("platform", {}).get("designation"),
                "country": pred.get("platform", {}).get("country"),
                "top_predictions": pred.get("top_predictions", []),
                "status": "success",
            })
        except Exception as e:
            results.append({"filename": filename, "error": str(e), "status": "failed"})

    return {
        "total": len(images),
        "successful": sum(1 for r in results if r.get("status") == "success"),
        "results": results,
    }



if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 55)
    print("  ASTRA VISION Engine running at http://127.0.0.1:8000")
    print("=" * 55 + "\n")
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
