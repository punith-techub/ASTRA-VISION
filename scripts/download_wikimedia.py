"""Optional public-data expansion with attribution saved for every downloaded file.

Example: python scripts/download_wikimedia.py --per-class 60
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import requests

QUERIES = {
    "aircraft": ["Category:Fighter aircraft", "Category:Military aircraft", "Category:Combat aircraft"],
    "helicopter": ["Category:Military helicopters", "Category:Attack helicopters"],
    "drone": ["Category:Unmanned aerial vehicles", "Category:Military unmanned aerial vehicles"],
    "military-vehicle": ["Category:Military vehicles", "Category:Tanks"],
    "naval": ["Category:Naval ships", "Category:Warships"],
}
API = "https://commons.wikimedia.org/w/api.php"
HEADERS = {"User-Agent": "ASTRA-VISION-training-project/1.0 (educational use)"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-class", type=int, default=40)
    parser.add_argument("--out", type=Path, default=Path("data/external/wikimedia"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = args.out.parent.parent / "manifests" / "wikimedia-credits.csv"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["category", "file", "title", "url", "license"])
        writer.writeheader()
        for category, category_sources in QUERIES.items():
            destination = args.out / category; destination.mkdir(parents=True, exist_ok=True)
            existing = list(destination.glob("*.*"))
            count = len(existing)
            if count >= args.per_class:
                print(f"{category}: already has {count} images, skipping")
                continue
            for cat_title in category_sources:
                if count >= args.per_class:
                    break
                params = {"action": "query", "generator": "categorymembers", "gcmtitle": cat_title,
                          "gcmtype": "file", "gcmlimit": (args.per_class - count) * 3, "prop": "imageinfo",
                          "iiprop": "url|extmetadata", "iiurlwidth": 960, "format": "json"}
                try:
                    response = requests.get(API, params=params, headers=HEADERS, timeout=30)
                    response.raise_for_status()
                except requests.RequestException:
                    continue
                for page in response.json().get("query", {}).get("pages", {}).values():
                    info = (page.get("imageinfo") or [{}])[0]
                    url = info.get("thumburl") or info.get("url")
                    if not url or count >= args.per_class: continue
                    suffix = Path(url.split("?")[0]).suffix.lower()
                    if suffix not in {".jpg", ".jpeg", ".png", ".webp"}: continue
                    filename = f"{count + 1:03d}-{page['pageid']}{suffix}"
                    target = destination / filename
                    if target.exists():
                        continue
                    try:
                        image = requests.get(url, headers=HEADERS, timeout=30); image.raise_for_status()
                        target.write_bytes(image.content)
                        metadata = info.get("extmetadata", {})
                        writer.writerow({"category": category, "file": str(target), "title": page.get("title", ""),
                                         "url": info.get("descriptionurl", url),
                                         "license": metadata.get("LicenseShortName", {}).get("value", "unknown")})
                        count += 1; time.sleep(0.15)
                    except requests.RequestException:
                        continue
            print(f"{category}: now has {count} images")


if __name__ == "__main__":
    main()
