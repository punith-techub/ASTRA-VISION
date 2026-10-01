"""ASTRA VISION — High-Throughput Batch Image Processor CLI.
Processes an entire directory of images, outputs classification, confidence, platform dossier,
and saves a comprehensive CSV/JSON report.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from PIL import Image

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from vision import VisionEngine, CLASSES, DISPLAY_NAMES


def run_batch(input_dir: Path, output_csv: Path | None, model_dir: Path) -> None:
    print(f"\n[ASTRA VISION] Initializing batch recognition engine from {model_dir}...")
    engine = VisionEngine(model_dir)

    exts = {".jpg", ".jpeg", ".png", ".webp"}
    files = [f for f in input_dir.iterdir() if f.is_file() and f.suffix.lower() in exts]

    if not files:
        print(f"No valid images found in {input_dir}.")
        return

    print(f"[ASTRA VISION] Found {len(files)} images to process in batch mode.")
    print("-" * 80)
    print(f"{'Filename':<32} | {'Category':<16} | {'Confidence':<10} | {'Platform':<20}")
    print("-" * 80)

    records = []
    t_start = time.time()

    for file_path in files:
        try:
            with Image.open(file_path) as img:
                pic = img.convert("RGB")
            res = engine.predict_local(pic, file_path.name)
            p_name = res.get("platform", {}).get("name", "Unknown")
            cat = res.get("label", res.get("prediction", "Unknown"))
            conf = res.get("confidence", 0.0)

            print(f"{file_path.name[:30]:<32} | {cat[:15]:<16} | {conf:<9}% | {p_name[:20]}")
            records.append({
                "filename": file_path.name,
                "category": res.get("prediction"),
                "category_label": cat,
                "confidence": conf,
                "platform_name": p_name,
                "country": res.get("platform", {}).get("country", ""),
                "year_origin": res.get("platform", {}).get("year_origin", ""),
                "status": "success",
            })
        except Exception as e:
            print(f"{file_path.name[:30]:<32} | ERROR: {str(e)[:35]}")
            records.append({
                "filename": file_path.name,
                "category": "error",
                "category_label": "error",
                "confidence": 0.0,
                "platform_name": "",
                "country": "",
                "year_origin": "",
                "status": f"failed: {e}",
            })

    t_total = time.time() - t_start
    print("-" * 80)
    print(f"[ASTRA VISION] Processed {len(records)} images in {t_total:.2f}s ({len(records)/max(t_total, 0.001):.1f} img/sec)")

    if output_csv:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
            writer.writeheader()
            writer.writerows(records)
        print(f"[ASTRA VISION] Batch summary exported to {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASTRA VISION Batch Image Processing CLI")
    parser.add_argument("--input-dir", "-i", type=Path, required=True, help="Directory containing images to process")
    parser.add_argument("--output-csv", "-o", type=Path, default=None, help="Path to save output CSV summary")
    parser.add_argument("--model-dir", "-m", type=Path, default=Path(__file__).resolve().parent.parent / "models")
    args = parser.parse_args()

    run_batch(args.input_dir, args.output_csv, args.model_dir)
