"""ASTRA VISION 4.0 — Multi-Object Tactical Detection & Optical Super-Resolution Zoom Engine.
Localizes up to 10+ distinct defense / aerospace / combat objects in a single frame.
Generates military HUD bounding box annotations, numbers each target [1], [2], [3]...,
performs super-resolution optical zoom crops for distant/blurred silhouettes,
and constructs individual tactical dossiers for every detected entity.
"""

from __future__ import annotations

import base64
import io
import math
from typing import Any
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
import defense_knowledge as dk

# Color palette for tactical HUD targets
HUD_COLORS = [
    {"hex": "#00ffcc", "rgb": (0, 255, 204), "bg": (0, 40, 35, 230), "name": "Tactical Cyan"},
    {"hex": "#ffb700", "rgb": (255, 183, 0), "bg": (45, 32, 0, 230), "name": "Radar Amber"},
    {"hex": "#00e676", "rgb": (0, 230, 118), "bg": (0, 42, 20, 230), "name": "Emerald Intercept"},
    {"hex": "#ff007f", "rgb": (255, 0, 127), "bg": (45, 0, 22, 230), "name": "Thermal Magenta"},
    {"hex": "#00bfff", "rgb": (0, 191, 255), "bg": (0, 32, 45, 230), "name": "Deep Sky Blue"},
    {"hex": "#e040fb", "rgb": (224, 64, 251), "bg": (38, 0, 45, 230), "name": "Neon Violet"},
    {"hex": "#76ff03", "rgb": (118, 255, 3), "bg": (20, 45, 0, 230), "name": "High-Vis Lime"},
]


def image_to_data_url(img: Image.Image, format: str = "JPEG", quality: int = 88) -> str:
    """Converts a PIL Image to a base64 data URL string."""
    buf = io.BytesIO()
    rgb = img.convert("RGB")
    rgb.save(buf, format=format, quality=quality)
    raw = buf.getvalue()
    b64 = base64.b64encode(raw).decode("utf-8")
    mime = "image/jpeg" if format.upper() == "JPEG" else "image/png"
    return f"data:{mime};base64,{b64}"


def compute_box_iou(boxA: tuple[int, int, int, int], boxB: tuple[int, int, int, int]) -> float:
    """Computes Intersection-over-Union (IoU) between two bounding boxes (ymin, xmin, ymax, xmax)."""
    yA = max(boxA[0], boxB[0])
    xA = max(boxA[1], boxB[1])
    yB = min(boxA[2], boxB[2])
    xB = min(boxA[3], boxB[3])
    interArea = max(0, yB - yA) * max(0, xB - xA)
    boxAArea = max(1, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = max(1, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))
    return interArea / float(boxAArea + boxBArea - interArea)


def snap_box_to_contour(
    img_arr: np.ndarray,
    box: tuple[int, int, int, int],  # (ymin, xmin, ymax, xmax)
    search_margin: int = 35,
) -> tuple[int, int, int, int]:
    """Snaps a rough bounding box to the true physical silhouette / contour boundaries.
    Uses gradient and luminance clustering in the local neighborhood.
    """
    try:
        from scipy import ndimage
        H, W = img_arr.shape[:2]
        ymin, xmin, ymax, xmax = box

        # Define neighborhood search window
        ny_min = max(0, ymin - search_margin)
        ny_max = min(H, ymax + search_margin)
        nx_min = max(0, xmin - search_margin)
        nx_max = min(W, xmax + search_margin)

        sub = img_arr[ny_min:ny_max, nx_min:nx_max]
        if sub.size == 0:
            return box

        # Detect salient silhouette or dark aircraft / vehicle pixels
        mean_lum = sub.mean(axis=2)
        bg_lum = np.median(mean_lum)
        
        # If background is bright (e.g. blue sky), foreground is darker than median
        if bg_lum > 110:
            foreground = mean_lum < (bg_lum - 20)
        else:
            foreground = mean_lum > (bg_lum + 25)

        labeled, n_features = ndimage.label(foreground)
        if n_features == 0:
            return box

        slices = ndimage.find_objects(labeled)
        best_slice = None
        best_area = 0

        # Target center in sub-coordinates
        sub_cy = (ymin + ymax) // 2 - ny_min
        sub_cx = (xmin + xmax) // 2 - nx_min

        for idx, sl in enumerate(slices, 1):
            area = int(np.sum(labeled[sl] == idx))
            if area > 40:
                ys, xs = sl
                # Prefer components close to the target center
                c_dist = math.hypot((ys.start + ys.stop) / 2 - sub_cy, (xs.start + xs.stop) / 2 - sub_cx)
                # Weighted score: large area + close to center
                score = area / (1.0 + c_dist * 0.1)
                if score > best_area:
                    best_area = score
                    best_slice = sl

        if best_slice:
            ys, xs = best_slice
            pad = 4  # small aesthetic padding
            snapped_ymin = max(0, ny_min + ys.start - pad)
            snapped_xmin = max(0, nx_min + xs.start - pad)
            snapped_ymax = min(H, ny_min + ys.stop + pad)
            snapped_xmax = min(W, nx_min + xs.stop + pad)
            return (snapped_ymin, snapped_xmin, snapped_ymax, snapped_xmax)
    except Exception:
        pass
    return box


def crop_and_super_sample(
    img: Image.Image,
    bbox_pixels: tuple[int, int, int, int],  # (ymin, xmin, ymax, xmax)
    target_dim: int = 360,
    pad_ratio: float = 0.22,
) -> tuple[str, bool, str]:
    """Crops an object with adaptive margin and applies optical super-sampling for distant targets.
    Returns: (crop_data_url, is_distant, zoom_factor_label)
    """
    W, H = img.size
    ymin, xmin, ymax, xmax = bbox_pixels

    bw = max(1, xmax - xmin)
    bh = max(1, ymax - ymin)

    # Determine if object is distant or small relative to the frame
    width_fraction = bw / W
    height_fraction = bh / H
    area_fraction = (bw * bh) / (W * H)

    is_distant = (width_fraction < 0.30) or (height_fraction < 0.30) or (area_fraction < 0.08)

    # Calculate padded box clamped to image edges
    pad_x = int(bw * pad_ratio)
    pad_y = int(bh * pad_ratio)

    crop_xmin = max(0, xmin - pad_x)
    crop_ymin = max(0, ymin - pad_y)
    crop_xmax = min(W, xmax + pad_x)
    crop_ymax = min(H, ymax + pad_y)

    crop = img.crop((crop_xmin, crop_ymin, crop_xmax, crop_ymax))

    # Calculate optical zoom factor
    max_orig_dim = max(crop.width, crop.height)
    if max_orig_dim < target_dim and is_distant:
        scale = max(2.0, min(6.0, target_dim / max(1, max_orig_dim)))
        zoom_w = int(crop.width * scale)
        zoom_h = int(crop.height * scale)
        zoomed = crop.resize((zoom_w, zoom_h), Image.Resampling.LANCZOS)

        # Enhance optical contrast & edge micro-structure for silhouettes
        enhancer = ImageEnhance.Sharpness(zoomed)
        zoomed = enhancer.enhance(1.8)
        contrast = ImageEnhance.Contrast(zoomed)
        zoomed = contrast.enhance(1.15)

        zoom_label = f"{round(scale, 1)}× Optical Zoom"
        return image_to_data_url(zoomed), True, zoom_label
    else:
        # Standard crop preview
        if max_orig_dim < 240:
            scale = 240 / max(1, max_orig_dim)
            crop = crop.resize((int(crop.width * scale), int(crop.height * scale)), Image.Resampling.LANCZOS)
        zoom_label = "1.0× Direct Optical"
        return image_to_data_url(crop), False, zoom_label


def generate_tactical_hud_image(
    original_img: Image.Image,
    targets: list[dict[str, Any]],
) -> str:
    """Renders a military-grade tactical HUD overlay with numbered bounding boxes,
    corner brackets, and platform labels onto the image.
    Returns: annotated_image_data_url
    """
    img = original_img.convert("RGB").copy()
    W, H = img.size
    draw = ImageDraw.Draw(img)

    occupied_label_regions = []

    for idx, t in enumerate(targets):
        color_info = HUD_COLORS[idx % len(HUD_COLORS)]
        hex_color = color_info["hex"]

        box = t.get("box_pixels")
        if not box or len(box) != 4:
            continue

        ymin, xmin, ymax, xmax = box
        bw = xmax - xmin
        bh = ymax - ymin

        # 1. Main outer bounding box (2px border)
        draw.rectangle([xmin, ymin, xmax, ymax], outline=hex_color, width=2)

        # 2. Corner brackets (L-shaped high-tech reticle corners)
        corner_len = max(6, min(16, int(min(bw, bh) * 0.30)))
        # Top-left corner
        draw.line([(xmin, ymin), (xmin + corner_len, ymin)], fill=hex_color, width=3)
        draw.line([(xmin, ymin), (xmin, ymin + corner_len)], fill=hex_color, width=3)
        # Top-right corner
        draw.line([(xmax, ymin), (xmax - corner_len, ymin)], fill=hex_color, width=3)
        draw.line([(xmax, ymin), (xmax, ymin + corner_len)], fill=hex_color, width=3)
        # Bottom-left corner
        draw.line([(xmin, ymax), (xmin + corner_len, ymax)], fill=hex_color, width=3)
        draw.line([(xmin, ymax), (xmin, ymax - corner_len)], fill=hex_color, width=3)
        # Bottom-right corner
        draw.line([(xmax, ymax), (xmax - corner_len, ymax)], fill=hex_color, width=3)
        draw.line([(xmax, ymax), (xmax, ymax - corner_len)], fill=hex_color, width=3)

        # 3. Center crosshair for precision targeting
        cx = (xmin + xmax) // 2
        cy = (ymin + ymax) // 2
        ch_len = 3
        draw.line([(cx - ch_len, cy), (cx + ch_len, cy)], fill=hex_color, width=1)
        draw.line([(cx, cy - ch_len), (cx, cy + ch_len)], fill=hex_color, width=1)

        # 4. Target header tag badge: [ID] Platform Name Conf%
        tid = t.get("id", idx + 1)
        name = t.get("exact_name", "Target")
        short_name = name.split("/")[0].strip()
        if len(short_name) > 20:
            short_name = short_name[:18] + "…"
        conf = t.get("confidence", 99.0)

        label_text = f"[{tid}] {short_name}  {conf}%"
        badge_w = min(220, max(110, len(label_text) * 8 + 12))
        badge_h = 20

        # Position badge above target if room, else below; stagger if overlaps prior label
        b_ymin = ymin - badge_h - 2 if ymin >= badge_h + 4 else ymax + 2
        b_ymax = b_ymin + badge_h
        b_xmin = xmin
        b_xmax = min(W - 2, xmin + badge_w)

        # Check overlap with prior labels
        for (oymin, oxmin, oymax, oxmax) in occupied_label_regions:
            if not (b_xmax < oxmin or b_xmin > oxmax or b_ymax < oymin or b_ymin > oymax):
                # Stagger vertically or shift right
                if b_ymin >= badge_h + 8:
                    b_ymin -= (badge_h + 4)
                    b_ymax = b_ymin + badge_h
                else:
                    b_ymin = ymax + 4
                    b_ymax = b_ymin + badge_h

        occupied_label_regions.append((b_ymin, b_xmin, b_ymax, b_xmax))

        # Draw dark backdrop pill
        draw.rectangle([b_xmin, b_ymin, b_xmax, b_ymax], fill="#041215", outline=hex_color, width=1)
        draw.text((b_xmin + 6, b_ymin + 3), label_text, fill=hex_color)

    return image_to_data_url(img, format="JPEG", quality=90)


def propose_candidate_regions_cv(
    img: Image.Image,
    max_proposals: int = 6,
) -> list[tuple[int, int, int, int]]:
    """Fast Computer Vision Region Proposer:
    Uses morphological operations and connected-component analysis to find distinct salient objects
    (e.g., aircraft in formation, vehicles, weapon systems) even when offline or as an ensemble fallback.
    Returns: list of (ymin, xmin, ymax, xmax) pixel coordinates.
    """
    try:
        from scipy import ndimage
        arr = np.array(img.convert("RGB"))
        H, W = arr.shape[:2]

        inner_mask = np.zeros((H, W), dtype=bool)
        m_y = max(4, int(H * 0.04))
        m_x = max(4, int(W * 0.04))
        inner_mask[m_y:H-m_y, m_x:W-m_x] = True

        is_dark = (arr.mean(axis=2) < 145) & inner_mask
        diff_r = np.abs(arr[:, :, 0] - arr[:, :, 2])
        is_silhouette = ((diff_r > 30) | is_dark) & inner_mask

        labeled, num_features = ndimage.label(is_silhouette)
        if num_features == 0:
            return []

        slices = ndimage.find_objects(labeled)
        candidates = []

        min_area = max(50, int((W * H) * 0.0008))
        max_area = int((W * H) * 0.85)

        for idx, sl in enumerate(slices, 1):
            ys, xs = sl
            area = np.sum(labeled[sl] == idx)
            if min_area <= area <= max_area:
                bw = xs.stop - xs.start
                bh = ys.stop - ys.start
                if bw >= 8 and bh >= 8:
                    candidates.append({
                        "area": area,
                        "box": (ys.start, xs.start, ys.stop, xs.stop),
                    })

        candidates.sort(key=lambda c: c["area"], reverse=True)
        return [c["box"] for c in candidates[:max_proposals]]
    except Exception as e:
        print(f"CV Region Proposer notice: {e}")
        return []


def decompose_wide_lineup_box(
    arr: np.ndarray,
    box: tuple[int, int, int, int],
    min_peaks: int = 2,
    max_peaks: int = 10,
) -> list[tuple[int, int, int, int]]:
    """Decomposes an elongated or multi-object bounding box (e.g. a row of cartridges/bullets
    or items standing side-by-side) into individual entity bounding boxes using column edge energy.
    """
    try:
        from scipy import ndimage
        from scipy.signal import find_peaks

        ymin, xmin, ymax, xmax = box
        bw = max(1, xmax - xmin)
        bh = max(1, ymax - ymin)

        # Only decompose if the box has a horizontal aspect ratio consistent with a lineup or row
        if bw < 40 or bh < 20 or (bw / float(bh) < 1.15 and bw < int(arr.shape[1] * 0.40)):
            return [box]

        sub = arr[ymin:ymax, xmin:xmax]
        sub_h, sub_w = sub.shape[:2]
        gray = sub.mean(axis=2).astype(float)

        gx = ndimage.sobel(gray, axis=1)
        gy = ndimage.sobel(gray, axis=0)
        grad = np.hypot(gx, gy)
        col_energy = np.sum(grad, axis=0)

        min_dist = max(10, int(sub_w * 0.04))
        prominence = col_energy.max() * 0.12
        peaks, _ = find_peaks(col_energy, distance=min_dist, prominence=prominence)

        if len(peaks) < min_peaks:
            return [box]

        if len(peaks) > max_peaks:
            peak_energies = [col_energy[p] for p in peaks]
            top_indices = np.argsort(peak_energies)[-max_peaks:]
            peaks = np.sort(peaks[top_indices])

        valleys = [0]
        for i in range(len(peaks) - 1):
            p1 = peaks[i]
            p2 = peaks[i + 1]
            v = p1 + int(np.argmin(col_energy[p1:p2+1]))
            valleys.append(v)
        valleys.append(sub_w)

        decomposed = []
        for i in range(len(valleys) - 1):
            x1, x2 = valleys[i], valleys[i + 1]
            seg_grad = grad[:, x1:x2]
            row_energy = np.sum(seg_grad, axis=1)
            thresh = row_energy.max() * 0.10
            active_y = np.where(row_energy > thresh)[0]
            if len(active_y) > 0:
                y1 = max(0, active_y[0] - 2)
                y2 = min(sub_h, active_y[-1] + 2)
            else:
                y1, y2 = 0, sub_h

            seg_col = col_energy[x1:x2]
            active_x = np.where(seg_col > seg_col.max() * 0.15)[0]
            if len(active_x) > 0:
                real_x1 = max(0, x1 + active_x[0] - 2)
                real_x2 = min(sub_w, x1 + active_x[-1] + 2)
            else:
                real_x1, real_x2 = x1, x2

            decomposed.append((
                ymin + y1,
                xmin + real_x1,
                ymin + y2,
                xmin + real_x2,
            ))

        return decomposed if len(decomposed) >= 2 else [box]
    except Exception:
        return [box]


def process_multi_target_recon(
    image: Image.Image,
    gemini_payload: dict[str, Any] | None,
    default_dossier: dict[str, Any] | None = None,
    crop_verifier: Any = None,
) -> dict[str, Any]:
    """Unifies spatial target detection, contour snapping, optical super-sampling zoom, and HUD rendering.
    Processes up to 10 targets in a single image.
    """
    img = image.convert("RGB")
    arr = np.array(img)
    W, H = img.size

    detected_targets: list[dict[str, Any]] = []

    # 1. Parse Gemini detected targets if available
    raw_targets = []
    if gemini_payload and isinstance(gemini_payload.get("detected_targets"), list):
        raw_targets = list(gemini_payload["detected_targets"])

    if not raw_targets and gemini_payload and gemini_payload.get("exact_name"):
        raw_targets = [gemini_payload]

    dossier = default_dossier or {}
    primary_name = dossier.get("name") or dossier.get("exact_name") or "Identified Object"

    # Lineup decomposition: if only 1 target was returned, check if it encompasses multiple distinct objects
    if len(raw_targets) == 1:
        single_t = raw_targets[0]
        raw_b = single_t.get("box_2d")
        if raw_b and len(raw_b) == 4:
            b_ymin = max(0, min(H - 1, int(raw_b[0] / 1000.0 * H)))
            b_xmin = max(0, min(W - 1, int(raw_b[1] / 1000.0 * W)))
            b_ymax = max(0, min(H, int(raw_b[2] / 1000.0 * H)))
            b_xmax = max(0, min(W, int(raw_b[3] / 1000.0 * W)))
            pixel_b = (b_ymin, b_xmin, b_ymax, b_xmax)
        else:
            pixel_b = (0, 0, H, W)

        decomposed_boxes = decompose_wide_lineup_box(arr, pixel_b)
        if len(decomposed_boxes) > 1:
            raw_targets = []
            for d_idx, d_box in enumerate(decomposed_boxes, 1):
                raw_targets.append({
                    "id": d_idx,
                    "box_2d": [
                        int(d_box[0] / H * 1000),
                        int(d_box[1] / W * 1000),
                        int(d_box[2] / H * 1000),
                        int(d_box[3] / W * 1000),
                    ],
                    "exact_name": f"{single_t.get('exact_name', primary_name)} #{d_idx}",
                    "designation": single_t.get("designation", "TARGET"),
                    "category": single_t.get("category", dossier.get("category", "ordnance")),
                    "category_label": single_t.get("category_label", dossier.get("category_label", "Munitions & Ordnance")),
                    "confidence": single_t.get("confidence", 95.0),
                })

    # Process each target
    for idx, rt in enumerate(raw_targets, 1):
        target_id = rt.get("id", idx)
        exact_name = rt.get("exact_name") or rt.get("name") or primary_name
        designation = rt.get("designation") or "—"
        cat = rt.get("category") or dossier.get("category", "object")
        cat_label = rt.get("category_label") or dossier.get("category_label", "Identified Object")
        conf = float(rt.get("confidence", dossier.get("confidence", 98.0)))

        # Coordinate conversion: box_2d [ymin, xmin, ymax, xmax] normalized on [0, 1000]
        raw_box = rt.get("box_2d")
        if raw_box and len(raw_box) == 4:
            ymin_px = max(0, min(H - 1, int(raw_box[0] / 1000.0 * H)))
            xmin_px = max(0, min(W - 1, int(raw_box[1] / 1000.0 * W)))
            ymax_px = max(0, min(H, int(raw_box[2] / 1000.0 * H)))
            xmax_px = max(0, min(W, int(raw_box[3] / 1000.0 * W)))
            if ymax_px <= ymin_px:
                ymax_px = min(H, ymin_px + 20)
            if xmax_px <= xmin_px:
                xmax_px = min(W, xmin_px + 20)
            initial_box = (ymin_px, xmin_px, ymax_px, xmax_px)
            # Refine and snap to true physical contour
            box_pixels = snap_box_to_contour(arr, initial_box)
            norm_box = [
                int(box_pixels[0] / H * 1000),
                int(box_pixels[1] / W * 1000),
                int(box_pixels[2] / H * 1000),
                int(box_pixels[3] / W * 1000),
            ]
        else:
            cv_boxes = propose_candidate_regions_cv(img, max_proposals=len(raw_targets))
            if cv_boxes and idx <= len(cv_boxes):
                box_pixels = cv_boxes[idx - 1]
            else:
                box_pixels = (int(H * 0.08), int(W * 0.08), int(H * 0.92), int(W * 0.92))
            norm_box = [
                int(box_pixels[0] / H * 1000),
                int(box_pixels[1] / W * 1000),
                int(box_pixels[2] / H * 1000),
                int(box_pixels[3] / W * 1000),
            ]

        # Extract super-sampled optical zoom crop
        crop_url, is_distant, zoom_label = crop_and_super_sample(img, box_pixels)

        # Resolve verified historical year and production count
        raw_year = rt.get("year_origin") or dossier.get("year_origin")
        raw_built = rt.get("number_built") or dossier.get("number_built")
        raw_op = rt.get("operational_count") or dossier.get("operational_count")
        t_year, t_built, t_op = dk.resolve_historical_metrics(
            name=exact_name,
            designation=designation,
            year_origin=raw_year,
            number_built=raw_built,
            operational_count=raw_op,
            category=cat,
            context_text=f"{rt.get('model_variant', '')} {rt.get('tactical_analysis', '')}",
        )

        target_obj = {
            "id": target_id,
            "exact_name": exact_name,
            "designation": designation,
            "model_variant": rt.get("model_variant", dossier.get("model_variant", "Standard Variant")),
            "category": cat,
            "category_label": cat_label,
            "confidence": round(conf, 1),
            "country": rt.get("country", dossier.get("country", "Global")),
            "manufacturer": rt.get("manufacturer", dossier.get("manufacturer", "Manufacturer")),
            "year_origin": t_year,
            "number_built": t_built,
            "operational_count": t_op,
            "role": rt.get("role", dossier.get("role", "Operations")),
            "status": rt.get("status", dossier.get("status", "Active")),
            "visual_signatures": rt.get("visual_signatures", dossier.get("visual_signatures", "Visual planform geometry.")),
            "suggested_search_query": rt.get("suggested_search_query", f"{exact_name} specifications"),
            "specs": rt.get("specs") or dossier.get("specs", {}),
            "box_2d": norm_box,
            "box_pixels": list(box_pixels),
            "crop_url": crop_url,
            "is_distant_target": is_distant,
            "zoom_factor": zoom_label,
            "tactical_analysis": rt.get("tactical_analysis", dossier.get("tactical_analysis", f"Target #{target_id} acquired in visual frame.")),
        }
        detected_targets.append(target_obj)

    # If no targets were found by multimodal vision, create a single clean target from the dominant object
    if not detected_targets:
        name = primary_name
        conf = float(dossier.get("confidence", 95.0))
        cv_boxes = propose_candidate_regions_cv(img, max_proposals=1)

        raw_year = dossier.get("year_origin")
        raw_built = dossier.get("number_built")
        raw_op = dossier.get("operational_count")
        t_year, t_built, t_op = dk.resolve_historical_metrics(
            name=name,
            designation=dossier.get("designation", ""),
            year_origin=raw_year,
            number_built=raw_built,
            operational_count=raw_op,
            category=dossier.get("category", "object"),
            context_text=f"{dossier.get('model_variant', '')} {dossier.get('tactical_analysis', '')}",
        )

        if cv_boxes:
            box = cv_boxes[0]
            crop_url, is_distant, zoom_label = crop_and_super_sample(img, box)
            norm_box = [
                int(box[0] / H * 1000),
                int(box[1] / W * 1000),
                int(box[2] / H * 1000),
                int(box[3] / W * 1000),
            ]
            detected_targets.append({
                "id": 1,
                "exact_name": name,
                "designation": dossier.get("designation", "—"),
                "model_variant": dossier.get("model_variant", "Standard Variant"),
                "category": dossier.get("category", "object"),
                "category_label": dossier.get("category_label", "Identified Object"),
                "confidence": round(conf, 1),
                "country": dossier.get("country", "Global"),
                "manufacturer": dossier.get("manufacturer", "Manufacturer"),
                "year_origin": t_year,
                "number_built": t_built,
                "operational_count": t_op,
                "role": dossier.get("role", "Operations"),
                "status": dossier.get("status", "Active"),
                "visual_signatures": dossier.get("visual_signatures", "Visual platform geometry."),
                "suggested_search_query": f"{name} specifications",
                "specs": dossier.get("specs", {}),
                "box_2d": norm_box,
                "box_pixels": list(box),
                "crop_url": crop_url,
                "is_distant_target": is_distant,
                "zoom_factor": zoom_label,
                "tactical_analysis": dossier.get("tactical_analysis", "Target acquired in visual sector."),
            })
        else:
            full_box = (int(H * 0.05), int(W * 0.05), int(H * 0.95), int(W * 0.95))
            crop_url, is_distant, zoom_label = crop_and_super_sample(img, full_box)
            detected_targets.append({
                "id": 1,
                "exact_name": name,
                "designation": dossier.get("designation", "—"),
                "model_variant": dossier.get("model_variant", "Standard Variant"),
                "category": dossier.get("category", "object"),
                "category_label": dossier.get("category_label", "Identified Object"),
                "confidence": round(conf, 1),
                "country": dossier.get("country", "Global"),
                "manufacturer": dossier.get("manufacturer", "Manufacturer"),
                "year_origin": t_year,
                "number_built": t_built,
                "operational_count": t_op,
                "role": dossier.get("role", "Operations"),
                "status": dossier.get("status", "Active"),
                "visual_signatures": dossier.get("visual_signatures", "Visual platform geometry."),
                "suggested_search_query": f"{name} specifications",
                "specs": dossier.get("specs", {}),
                "box_2d": [50, 50, 950, 950],
                "box_pixels": list(full_box),
                "crop_url": crop_url,
                "is_distant_target": False,
                "zoom_factor": "1.0× Direct Optical",
                "tactical_analysis": dossier.get("tactical_analysis", "Target acquired in optical sector."),
            })

    # Optical Crop Disambiguation:
    # If multiple targets are present and any share identical base names (e.g. side-by-side comparison images where the model copied the label),
    # verify each target's optical crop independently to assign its genuine make, model, and specifications.
    if crop_verifier and len(detected_targets) >= 2:
        import re

        def normalize_target_name(n: str) -> str:
            # Strip any parenthetical suffix such as (Dark Grey Scheme), (Left), (Geometric Digital Camouflage), etc.
            cleaned = re.sub(r"\s*\([^)]*\)", "", n)
            return cleaned.strip().lower()

        base_names = [normalize_target_name(t.get("exact_name", "")) for t in detected_targets]
        has_duplicates = len(base_names) > len(set(base_names))
        if has_duplicates:
            for idx, t in enumerate(detected_targets):
                b_name = base_names[idx]
                if base_names.count(b_name) > 1:
                    box_px = t.get("box_pixels")
                    if box_px and len(box_px) == 4:
                        sub_crop = img.crop((box_px[1], box_px[0], box_px[3], box_px[2]))
                        verified = crop_verifier(sub_crop)
                        if verified and verified.get("exact_name"):
                            v_name = verified["exact_name"]
                            if "Identified Defense Platform" not in v_name:
                                t["exact_name"] = v_name
                                if verified.get("designation"):
                                    t["designation"] = verified["designation"]
                                if verified.get("category"):
                                    t["category"] = verified["category"]
                                if verified.get("category_label"):
                                    t["category_label"] = verified["category_label"]
                                if verified.get("country"):
                                    t["country"] = verified["country"]
                                if verified.get("manufacturer"):
                                    t["manufacturer"] = verified["manufacturer"]
                                if verified.get("year_origin"):
                                    t["year_origin"] = verified["year_origin"]
                                if verified.get("number_built"):
                                    t["number_built"] = verified["number_built"]
                                if verified.get("operational_count"):
                                    t["operational_count"] = verified["operational_count"]
                                if verified.get("specs"):
                                    t["specs"] = verified["specs"]
                                if verified.get("visual_signatures"):
                                    t["visual_signatures"] = verified["visual_signatures"]
                                if verified.get("tactical_analysis"):
                                    t["tactical_analysis"] = verified["tactical_analysis"]

                                t_yr, t_bt, t_op = dk.resolve_historical_metrics(
                                    name=t["exact_name"],
                                    designation=t.get("designation", ""),
                                    year_origin=t.get("year_origin"),
                                    number_built=t.get("number_built"),
                                    operational_count=t.get("operational_count"),
                                    category=t.get("category", "object"),
                                    context_text=f"{t.get('model_variant', '')} {t.get('tactical_analysis', '')}",
                                )
                                t["year_origin"] = t_yr
                                t["number_built"] = t_bt
                                t["operational_count"] = t_op

    # Render tactical military HUD annotated image
    annotated_image_url = generate_tactical_hud_image(img, detected_targets)

    return {
        "target_count": len(detected_targets),
        "detected_targets": detected_targets,
        "annotated_image_url": annotated_image_url,
        "has_distant_targets": any(t["is_distant_target"] for t in detected_targets),
    }
