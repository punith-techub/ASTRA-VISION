"""ASTRA VISION — Optional Gemini Multimodal Tactical Intelligence Connector.
Provides open-ended deep tactical reconnaissance for rare, foreign, or modified military platforms.
If no API key is provided or the service is unreachable, the system gracefully falls back
to the local offline CLIP + Defense Knowledge Base engine.
"""

from __future__ import annotations

import base64
import io
import json
import os
from typing import Any

from dotenv import load_dotenv
from PIL import Image
import requests

load_dotenv()

CANDIDATE_MODELS = [
    "gemini-3-flash-preview",
    "gemini-flash-latest",
    "gemini-flash-lite-latest",
    "gemini-2.5-flash",
]
DEFAULT_MODEL = "gemini-3-flash-preview"
API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


RECON_PROMPT = """You are an elite military intelligence reconnaissance and defense systems expert.
Analyze this military defense platform image with extreme precision and domain expertise.
Identify the exact platform, variant, country, historical data, and observable tactical details.

You must respond with ONLY a valid JSON object strictly matching this schema:
{
  "exact_name": "Full official manufacturer and model name (e.g. Platform / Entity Name)",
  "designation": "Operational designation or code (e.g. Model / Mark)",
  "model_variant": "Specific model or block variant",
  "category": "aircraft" | "helicopter" | "drone" | "military-vehicle" | "naval",
  "category_label": "Fighter Aircraft" | "Helicopter" | "Drone / UAV" | "Military Vehicle" | "Naval Vessel",
  "country": "Country of origin with flag emoji",
  "manufacturer": "Design bureau or prime contractor",
  "year_origin": "Year of first flight and year of commissioning/induction (e.g. 1997 / 2005)",
  "number_built": "Total production volume (e.g. 195 aircraft built)",
  "operational_count": "Estimated active operational count in service today (e.g. ~183 in USAF active inventory)",
  "status": "Current operational status",
  "role": "Operational tactical classification",
  "specs": {
    "speed": "Top speed and cruise speed",
    "combat_range": "Combat radius / ferry range",
    "ceiling": "Service ceiling / maximum altitude / operating depth",
    "armament": "Weapons loadout visible or standard combat configuration",
    "avionics": "Sensors, radar arrays, and electronic warfare suite"
  },
  "visual_signatures": "Key visual recognition features that identify this specific platform in this photo",
  "tactical_analysis": "2-3 sentences of sharp tactical reconnaissance analysis of this specific image (observable livery, ordnance, theater setting, or visual cues)"
}
"""


DEFAULT_KEY = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or ""


def is_gemini_available(custom_key: str | None = None) -> bool:
    key = custom_key or DEFAULT_KEY or os.environ.get("GOOGLE_API_KEY")
    return bool(key and key.strip())


def analyze_with_gemini(
    image: Image.Image,
    api_key: str | None = None,
    timeout_sec: int = 20,
) -> dict[str, Any] | None:
    """Invokes Google Gemini 2.5 Flash with the image to perform deep tactical reconnaissance."""
    key = api_key or DEFAULT_KEY or os.environ.get("GOOGLE_API_KEY")
    if not key or not key.strip():
        return None

    # Resize if large to ensure sub-second transmission
    img = image.convert("RGB")
    max_dim = 1024
    if max(img.size) > max_dim:
        ratio = max_dim / max(img.size)
        new_size = (int(img.width * ratio), int(img.height * ratio))
        img = img.resize(new_size, Image.Resampling.LANCZOS)

    # Convert image to compressed JPEG buffer
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    for model_name in CANDIDATE_MODELS:
        url = API_URL.format(model=model_name) + f"?key={key.strip()}"
        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": RECON_PROMPT},
                        {
                            "inline_data": {
                                "mime_type": "image/jpeg",
                                "data": img_b64,
                            }
                        },
                    ]
                }
            ],
            "generationConfig": {
                "response_mime_type": "application/json",
                "temperature": 0.2,
            },
        }

        try:
            response = requests.post(url, headers=headers, json=payload, timeout=timeout_sec)
            if response.status_code == 200:
                result_json = response.json()
                candidates = result_json.get("candidates", [])
                if candidates:
                    text = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "")
                    if text:
                        return json.loads(text.strip())
            print(f"Gemini Tactical ({model_name}) status {response.status_code}")
        except Exception as e:
            print(f"Gemini Tactical ({model_name}) note: {e}")

    return None
