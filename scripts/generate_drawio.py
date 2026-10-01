#!/usr/bin/env python3
"""Generates a professional, beautiful draw.io XML diagram for ASTRA-VISION 5.0."""

import xml.etree.ElementTree as ET
from pathlib import Path


def create_drawio_xml() -> str:
    root = ET.Element("mxfile", {
        "host": "app.diagrams.net",
        "modified": "2026-10-01T21:40:00.000Z",
        "agent": "ASTRA-VISION Architect",
        "version": "24.0.0",
        "type": "device",
    })

    diagram = ET.SubElement(root, "diagram", {
        "id": "astra-vision-arch",
        "name": "ASTRA-VISION 5.0 Architecture",
    })

    model = ET.SubElement(diagram, "mxGraphModel", {
        "dx": "1600",
        "dy": "1000",
        "grid": "1",
        "gridSize": "10",
        "guides": "1",
        "tooltips": "1",
        "connect": "1",
        "arrows": "1",
        "fold": "1",
        "page": "1",
        "pageScale": "1",
        "pageWidth": "1650",
        "pageHeight": "1150",
        "background": "#0b0f19",
        "math": "0",
        "shadow": "0",
    })

    root_cells = ET.SubElement(model, "root")
    ET.SubElement(root_cells, "mxCell", {"id": "0"})
    ET.SubElement(root_cells, "mxCell", {"id": "1", "parent": "0"})

    cells = []

    def add_cell(id_str, val, style, x, y, w, h, parent="1"):
        c = ET.SubElement(root_cells, "mxCell", {
            "id": id_str,
            "value": val,
            "style": style,
            "parent": parent,
            "vertex": "1",
        })
        ET.SubElement(c, "mxGeometry", {
            "x": str(x),
            "y": str(y),
            "width": str(w),
            "height": str(h),
            "as": "geometry",
        })
        return c

    def add_edge(id_str, src, trg, style, val="", parent="1", points=None):
        e = ET.SubElement(root_cells, "mxCell", {
            "id": id_str,
            "value": val,
            "style": style,
            "parent": parent,
            "edge": "1",
            "source": src,
            "target": trg,
        })
        geo = ET.SubElement(e, "mxGeometry", {
            "relative": "1",
            "as": "geometry",
        })
        if points:
            arr = ET.SubElement(geo, "Array", {"as": "points"})
            for px, py in points:
                ET.SubElement(arr, "mxPoint", {"x": str(px), "y": str(py)})
        return e

    # =========================================================================
    # STYLES
    # =========================================================================
    TITLE_STYLE = "text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;whiteSpace=wrap;rounded=0;fontColor=#f8fafc;fontSize=20;fontStyle=1;"
    SUBTITLE_STYLE = "text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;whiteSpace=wrap;rounded=0;fontColor=#94a3b8;fontSize=12;fontStyle=0;"

    CONTAINER_STYLE = "swimlane;whiteSpace=wrap;html=1;rounded=1;arcSize=8;fontColor=#f8fafc;fontStyle=1;fontSize=13;strokeWidth=1.5;shadow=1;"
    CONTAINER_UI = f"{CONTAINER_STYLE}fillColor=#0f172a;strokeColor=#6366f1;startSize=30;"
    CONTAINER_API = f"{CONTAINER_STYLE}fillColor=#0f172a;strokeColor=#06b6d4;startSize=30;"
    CONTAINER_ML = f"{CONTAINER_STYLE}fillColor=#0f172a;strokeColor=#a855f7;startSize=30;"
    CONTAINER_LEARN = f"{CONTAINER_STYLE}fillColor=#0f172a;strokeColor=#ec4899;startSize=30;"
    CONTAINER_FUTURE = f"{CONTAINER_STYLE}fillColor=#111827;strokeColor=#64748b;dashed=1;strokeWidth=1.5;startSize=30;"

    BOX_DARK = "rounded=1;arcSize=8;whiteSpace=wrap;html=1;fillColor=#1e293b;strokeColor=#475569;fontColor=#f8fafc;fontSize=11;strokeWidth=1;"
    BOX_UI = "rounded=1;arcSize=8;whiteSpace=wrap;html=1;fillColor=#1e1b4b;strokeColor=#818cf8;fontColor=#ffffff;fontSize=11;strokeWidth=1.5;"
    BOX_API = "rounded=1;arcSize=8;whiteSpace=wrap;html=1;fillColor=#164e63;strokeColor=#22d3ee;fontColor=#ffffff;fontSize=11;strokeWidth=1.5;"
    BOX_ML = "rounded=1;arcSize=8;whiteSpace=wrap;html=1;fillColor=#3b0764;strokeColor=#c084fc;fontColor=#ffffff;fontSize=11;strokeWidth=1.5;"
    BOX_DB = "rounded=1;arcSize=8;whiteSpace=wrap;html=1;fillColor=#064e3b;strokeColor=#34d399;fontColor=#ffffff;fontSize=11;strokeWidth=1.5;"
    BOX_OFFLINE = "rounded=1;arcSize=8;whiteSpace=wrap;html=1;fillColor=#78350f;strokeColor=#fbbf24;fontColor=#ffffff;fontSize=11;fontStyle=1;strokeWidth=2;"
    BOX_REWARD = "rounded=1;arcSize=8;whiteSpace=wrap;html=1;fillColor=#065f46;strokeColor=#10b981;fontColor=#ffffff;fontSize=11;strokeWidth=1.5;"
    BOX_PUNISH = "rounded=1;arcSize=8;whiteSpace=wrap;html=1;fillColor=#881337;strokeColor=#fb7185;fontColor=#ffffff;fontSize=11;strokeWidth=1.5;"
    BOX_FUTURE = "rounded=1;arcSize=8;whiteSpace=wrap;html=1;fillColor=#18181b;strokeColor=#94a3b8;fontColor=#e2e8f0;fontSize=11;dashed=1;strokeWidth=1.5;"

    EDGE_STD = "edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;html=1;strokeColor=#94a3b8;strokeWidth=1.5;fontSize=10;fontColor=#cbd5e1;"
    EDGE_OFFLINE = "edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;html=1;strokeColor=#fbbf24;strokeWidth=2;dashed=1;fontSize=10;fontColor=#fde68a;"
    EDGE_REWARD = "edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;html=1;strokeColor=#10b981;strokeWidth=2;fontSize=10;fontColor=#6ee7b7;"
    EDGE_PUNISH = "edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;html=1;strokeColor=#f43f5e;strokeWidth=2;fontSize=10;fontColor=#fda4af;"

    # =========================================================================
    # HEADER / TITLE
    # =========================================================================
    add_cell("title", "ASTRA-VISION 5.0 — FULL SYSTEM ARCHITECTURE", TITLE_STYLE, 40, 20, 1560, 30)
    add_cell("subtitle", "Offline-First Dual-Model Ensemble &bull; Explainable Grad-CAM &bull; Multi-Target Engine &bull; Two-Stage Online Self-Learning", SUBTITLE_STYLE, 40, 50, 1560, 20)

    # =========================================================================
    # 1. CLIENT LAYER (Left: X=40, W=260)
    # =========================================================================
    add_cell("c_ui", "1. CLIENT LAYER — TACTICAL HUD", CONTAINER_UI, 40, 85, 270, 715)
    add_cell("ui_user", "<b>Operator / User</b><br>(Field Analyst, Drone Operator)", BOX_DARK, 60, 125, 230, 45)
    add_cell("ui_upload", "<b>Image & Batch Ingestion</b><br>&bull; Drag & Drop / Single File<br>&bull; Batch Modal (up to 30 imgs)<br>&bull; JPG, PNG, WebP supported", BOX_UI, 60, 190, 230, 60)
    add_cell("ui_hud", "<b>Tactical HUD Display</b><br>&bull; Primary Military Category<br>&bull; Calibrated Confidence %<br>&bull; Top-3 Category Breakdown<br>&bull; Technical Intelligence Dossier", BOX_UI, 60, 270, 230, 75)
    add_cell("ui_cam_toggle", "<b>Grad-CAM Saliency Toggle</b><br>&bull; ColorJet Visual Heatmap<br>&bull; Overlays aerodynamic wings,<br>rotors, or armor turrets", BOX_UI, 60, 365, 230, 65)
    add_cell("ui_multitarget", "<b>Multi-Target Selector Tabs</b><br>&bull; Target 1 ... Target 10<br>&bull; Optical Bounding Boxes<br>&bull; Independent Platform Dossiers", BOX_UI, 60, 450, 230, 65)
    add_cell("ui_offline_badge", "<b>Offline Status Badge</b><br>&bull; &apos;OFFLINE MODE (LOCAL MODEL)&apos;<br>&bull; Instant on-device indication<br>&bull; Zero UI freeze / lag", BOX_OFFLINE, 60, 535, 230, 65)
    add_cell("ui_history_metrics", "<b>Scan History & Metrics</b><br>&bull; LocalStorage Scan History<br>&bull; Interactive ROC-AUC Curves (0.9898)<br>&bull; Confusion Matrix Display", BOX_UI, 60, 620, 230, 65)
    add_cell("ui_batch_csv", "<b>Batch CSV Export</b><br>&bull; Structured tabular metrics<br>&bull; Standalone CLI script", BOX_DARK, 60, 705, 230, 45)

    # =========================================================================
    # 2. BACKEND API & PREPROCESSING LAYER (X=340, W=260)
    # =========================================================================
    add_cell("c_api", "2. API & MULTI-OBJECT ENGINE", CONTAINER_API, 340, 85, 260, 715)
    add_cell("api_fastapi", "<b>FastAPI Asynchronous Gateway</b><br><code>app.py (Uvicorn Port 8000)</code><br>&bull; /api/predict<br>&bull; /api/predict/batch<br>&bull; /api/metrics & /api/platforms", BOX_API, 355, 125, 230, 75)
    add_cell("api_sanitizer", "<b>Input Sanitizer & Guardrails</b><br>&bull; 10 MB Max File Limit (413)<br>&bull; Format Check (415 for non-image)<br>&bull; Empty / Corrupt Catch (400)", BOX_DARK, 355, 220, 230, 65)
    add_cell("api_multiobj", "<b>Multi-Object Engine</b><br><code>multi_object_engine.py</code><br>&bull; Contour & Edge Detector<br>&bull; Lineup & Formation Separator<br>&bull; Bounding Box Coordinates<br>&bull; Sub-Pixel High-Res Crops", BOX_API, 355, 305, 230, 95)
    add_cell("api_preprocess", "<b>Image Preprocessor</b><br>&bull; Aspect-ratio preservation<br>&bull; Bicubic Resize & Center Crop<br>&bull; ImageNet Normalization<br>&bull; Monochrome Autocontrast", BOX_DARK, 355, 420, 230, 75)

    # Network Gateway Diamond / Router
    add_cell("gw_socket_check", "<b>Network Liveness Check</b><br><code>is_internet_available()</code><br>Socket ping 1.1.1.1 / 8.8.8.8<br>(0.8s timeout, 5s cache)", "rhombus;whiteSpace=wrap;html=1;fillColor=#1e293b;strokeColor=#38bdf8;fontColor=#ffffff;fontSize=10;fontStyle=1;", 380, 525, 180, 110)
    add_cell("api_response_builder", "<b>Unified Tactical Response</b><br>&bull; Bounding boxes + Crops<br>&bull; Heatmap base64 data<br>&bull; Dossier + Specs + Weapons<br>&bull; Health score & learning log", BOX_API, 355, 665, 230, 75)

    # =========================================================================
    # 3. LOCAL NEURAL VISION PIPELINE (Center: X=630, W=400)
    # =========================================================================
    add_cell("c_ml", "3. LOCAL NEURAL VISION PIPELINE (ON-DEVICE CPU)", CONTAINER_ML, 630, 85, 400, 715)
    add_cell("ml_mem_check", "<b>Online Memory Exemplar Query</b><br><code>online_memory.pt (512-D vectors)</code><br>Checks cosine similarity (&gt; 0.82) to match<br>previously learned corrected images first!", BOX_DB, 650, 125, 360, 60)

    # Stage 1: Category Classification
    add_cell("stg1_box", "<b>STAGE 1: DUAL-MODEL ENSEMBLE CLASSIFICATION</b>", "swimlane;whiteSpace=wrap;html=1;fillColor=#1e1b4b;strokeColor=#a855f7;fontColor=#ffffff;fontSize=11;fontStyle=1;startSize=24;rounded=1;", 650, 205, 360, 140)
    add_cell("stg1_mobilenet", "<b>MobileNetV3-Small</b><br>Fine-Tuned CNN<br>&bull; 18ms CPU latency<br>&bull; Spatial edge detection", BOX_ML, 665, 238, 155, 60, parent="stg1_box")
    add_cell("stg1_clip_txt", "<b>OpenAI CLIP ViT-B/32</b><br>Zero-Shot Vision-Lang<br>&bull; Multi-prompt ensemble<br>&bull; Broad semantic context", BOX_ML, 835, 238, 160, 60, parent="stg1_box")
    add_cell("stg1_fusion", "<b>50 / 50 Probability Fusion</b> &rarr; <b>90.38% Acc | 0.9898 ROC-AUC</b>", "rounded=1;arcSize=8;whiteSpace=wrap;html=1;fillColor=#4c1d95;strokeColor=#c084fc;fontColor=#ffffff;fontSize=10;fontStyle=1;", 665, 305, 330, 30, parent="stg1_box")

    # Stage 2: Fine-Grained Platform Matching
    add_cell("stg2_box", "<b>STAGE 2: FINE-GRAINED PLATFORM MATCHING</b>", "swimlane;whiteSpace=wrap;html=1;fillColor=#064e3b;strokeColor=#10b981;fontColor=#ffffff;fontSize=11;fontStyle=1;startSize=24;rounded=1;", 650, 365, 360, 150)
    add_cell("stg2_vec", "<b>512-D Visual Feature Vector</b><br>Extracted via CLIP Image Encoder", BOX_ML, 665, 398, 155, 55, parent="stg2_box")
    add_cell("stg2_kb", "<b>Defense Knowledge Base</b><br>74+ Platforms (Specs & Weapons)<br><code>platform_embeddings.pt</code>", BOX_DB, 835, 398, 160, 55, parent="stg2_box")
    add_cell("stg2_matcher", "<b>Cosine Similarity Matcher + Calibrated Confidence</b><br>Matches visual vector against platform embeddings<br>Applies margin penalty if decision is tight (&lt;0.008)", BOX_DB, 665, 460, 330, 45, parent="stg2_box")

    # Stage 3: Explainable AI (Grad-CAM)
    add_cell("stg3_box", "<b>STAGE 3: EXPLAINABLE AI (Grad-CAM)</b>", "swimlane;whiteSpace=wrap;html=1;fillColor=#1e293b;strokeColor=#38bdf8;fontColor=#ffffff;fontSize=11;fontStyle=1;startSize=24;rounded=1;", 650, 535, 360, 100)
    add_cell("stg3_hook", "<b>PyTorch Backward Hook</b><br>Attached to <code>features[-1]</code> layer<br>of MobileNetV3 to extract gradients", BOX_DARK, 665, 568, 160, 55, parent="stg3_box")
    add_cell("stg3_heatmap", "<b>ColorJet Saliency Heatmap</b><br>Averages positive gradients &amp;<br>generates direct visual overlay", BOX_API, 835, 568, 160, 55, parent="stg3_box")

    add_cell("ml_calibrated_dossier", "<b>Complete Local Intelligence Dossier</b><br>&bull; Category & Top-3 Candidates &bull; Exact Model Variant &bull; Manufacturer &bull; Year<br>&bull; Operational & Fleet Count &bull; Combat Speed, Range & Armament", BOX_DARK, 650, 655, 360, 65)

    # =========================================================================
    # 4. EXECUTION ROUTING: OFFLINE MODE vs ONLINE SELF-LEARNING (X=1060, W=260)
    # =========================================================================
    add_cell("c_learn", "4. VERIFICATION & OFFLINE ROUTING", CONTAINER_LEARN, 1060, 85, 260, 715)

    # OFFLINE MODE BOX
    add_cell("off_mode_box", "<b>OFFLINE MODE (LATEST V5)</b><br><i>When Network is Jammed / Offline:</i><br>&bull; External APIs bypassed 100%<br>&bull; Zero network latency / timeout<br>&bull; Immediate local model inference<br>&bull; Fully calibrated local confidence<br>&bull; Sets <code>is_offline=True</code> badge", BOX_OFFLINE, 1075, 125, 230, 115)

    # ONLINE MODE BOX
    add_cell("on_mode_title", "<b>ONLINE VERIFICATION MODE</b><br><i>When Internet is Available:</i>", "text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;fontColor=#f472b6;fontSize=11;fontStyle=1;", 1075, 255, 230, 20)
    add_cell("on_ai_check", "<b>External Multimodal AI Check</b><br>&bull; Google Gemini 2.5 Flash<br>&bull; OpenRouter / Groq LPU<br>&bull; Synthesizes cross-model consensus", BOX_DARK, 1075, 280, 230, 75)
    add_cell("on_judge", "<b>Consensus Judge & Evaluation</b><br><code>online_trainer.py</code><br>Compares Local Guess vs Ground Truth", "rhombus;whiteSpace=wrap;html=1;fillColor=#1e293b;strokeColor=#f472b6;fontColor=#ffffff;fontSize=10;fontStyle=1;", 1100, 375, 180, 85)

    add_cell("on_reward", "<b>MATCH: REWARD FLOW</b><br>&bull; +10 Reward Points<br>&bull; Health score increased<br>&bull; Confirms local answer", BOX_REWARD, 1075, 480, 230, 65)

    add_cell("on_punish", "<b>MISMATCH: SEVERE PUNISHMENT</b><br>&bull; <b>-50 Penalty Points</b> deducted<br>&bull; Loss Penalty (4x multiplier)<br>&bull; Immediate Adam gradient step on<br>MobileNet classification head<br>&bull; <b>Memorizes 512-D vector into<br><code>online_memory.pt</code></b>", BOX_PUNISH, 1075, 560, 230, 110)

    add_cell("on_mem_store", "<b>Persistent Neural Memory</b><br>Retains learned exemplars across restarts", BOX_DB, 1075, 685, 230, 45)

    # =========================================================================
    # 5. DATA & MODEL ASSETS (X=1350, W=260)
    # =========================================================================
    add_cell("c_assets", "5. PERSISTENT ASSETS & MODELS", CONTAINER_STYLE + "fillColor=#0f172a;strokeColor=#10b981;startSize=30;", 1350, 85, 260, 715)
    add_cell("ast_mnet", "<b>mobilenet_v3_small.pt</b><br>Fine-tuned PyTorch CNN weights<br>(Trained with CosineAnnealingLR)", BOX_DARK, 1365, 125, 230, 60)
    add_cell("ast_clip", "<b>CLIP ViT-B/32 Weights</b><br>HuggingFace Transformers Cache<br>(Zero-shot vision & language)", BOX_DARK, 1365, 200, 230, 60)
    add_cell("ast_kb_embeds", "<b>platform_embeddings.pt</b><br>Precomputed 512-D vector matrix<br>for 74+ military platforms", BOX_DB, 1365, 275, 230, 65)
    add_cell("ast_defense_kb", "<b>defense_knowledge.py</b><br>Structured intelligence database:<br>country, year, fleet, role, specs", BOX_DB, 1365, 355, 230, 65)
    add_cell("ast_online_mem", "<b>online_memory.pt</b><br>Dynamic neural memory bank of<br>corrected/penalized exemplars", BOX_DB, 1365, 435, 230, 65)
    add_cell("ast_metrics_json", "<b>metrics.json & Plots</b><br>Precomputed ROC curves &<br>confusion matrix evaluation plots", BOX_DARK, 1365, 515, 230, 60)
    add_cell("ast_dataset", "<b>Curated Dataset (258 Imgs)</b><br>&bull; ~150 Challenge starter images<br>&bull; 108 Wikimedia open-license images<br>&bull; 80% Train / 20% Val stratified split", BOX_DARK, 1365, 590, 230, 75)
    add_cell("ast_tests", "<b>Automated PyTest Suite</b><br>14 unit & integration tests<br>100% pass rate in 91 seconds", BOX_REWARD, 1365, 680, 230, 50)

    # =========================================================================
    # 6. FUTURE PLANNED EXTENSIONS (Bottom: Full width Y=820, H=280)
    # =========================================================================
    add_cell("c_future", "6. FUTURE PLANNED MULTIMODAL & OSINT EXTENSIONS (ROADMAP)", CONTAINER_FUTURE, 40, 825, 1570, 260)

    add_cell("fut_sound", "<b>1. Sound-Based Prediction (Acoustic AI)</b><br>&bull; Predicts platform identity using acoustic audio signatures<br>&bull; Matches jet turbine roar, rotor blade harmonics, and artillery blast frequencies<br>&bull; Complements visual detection in heavy fog, darkness, or tree canopies", BOX_FUTURE, 60, 870, 360, 95, parent="c_future")

    add_cell("fut_live", "<b>2. Live Stream Mode (Gemini Live Style)</b><br>&bull; Continuous real-time camera feed analysis (WebRTC / RTSP)<br>&bull; Point camera at sky/field &rarr; continuous tracking & object detection<br>&bull; Instant bounding box tracking (ByteTrack) with live HUD tactical telemetry overlay", BOX_FUTURE, 440, 870, 360, 95, parent="c_future")

    add_cell("fut_text", "<b>3. Natural Language Feature Query Input</b><br>&bull; Operator inputs observed physical features via text input<br>&bull; Example: <i>&apos;Delta wings, twin canted fins, dual engine, refueling probe&apos;</i><br>&bull; Model queries 512-D vector space &amp; defense DB to predict matching platforms", BOX_FUTURE, 820, 870, 360, 95, parent="c_future")

    add_cell("fut_osint", "<b>4. Contextual OSINT & Flight Radar Scraping</b><br>&bull; Real-world situational awareness from text sightings<br>&bull; Example: User says <i>&apos;I saw a fighter jet today over my house in Bangalore&apos;</i><br>&bull; System scrapes real-time news, airport NOTAMs, ADS-B &amp; Flightradar tracking<br>&bull; Detects local sorties (HAL Tejas / Su-30MKI) and presents matching aircraft &amp; specs", BOX_FUTURE, 1200, 870, 390, 95, parent="c_future")

    add_cell("fut_banner", "<b>Architecture Scalability Note:</b> The modular decoupled pipeline allows acoustic audio encoders, live RTSP frame grabbers, and web scraping workers to plug directly into the Unified Response Builder without restructuring the core local engine.", "text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;fontColor=#94a3b8;fontSize=11;fontStyle=2;", 60, 980, 1530, 25, parent="c_future")

    # =========================================================================
    # EDGES & CONNECTIONS
    # =========================================================================
    # User -> Upload -> API
    add_edge("e1", "ui_user", "ui_upload", EDGE_STD)
    add_edge("e2", "ui_upload", "api_fastapi", EDGE_STD, val="HTTP POST /api/predict")

    # API -> Sanitizer -> Multi-object -> Preprocess
    add_edge("e3", "api_fastapi", "api_sanitizer", EDGE_STD)
    add_edge("e4", "api_sanitizer", "api_multiobj", EDGE_STD, val="Valid image")
    add_edge("e5", "api_multiobj", "api_preprocess", EDGE_STD, val="Target crops")

    # Preprocess -> Local Vision Pipeline
    add_edge("e6", "api_preprocess", "ml_mem_check", EDGE_STD, val="Normalized Tensor")

    # Mem Check -> Stage 1 & 2
    add_edge("e7", "ml_mem_check", "stg1_box", EDGE_STD, val="New Image")
    add_edge("e8", "stg1_box", "stg2_box", EDGE_STD, val="Category probs")
    add_edge("e9", "stg1_box", "stg3_box", EDGE_STD, val="Predicted class index")
    add_edge("e10", "stg2_box", "ml_calibrated_dossier", EDGE_STD)
    add_edge("e11", "stg3_box", "ml_calibrated_dossier", EDGE_STD, val="Heatmap overlay")

    # Local Result -> Network Gateway Socket Check
    add_edge("e12", "ml_calibrated_dossier", "gw_socket_check", EDGE_STD, val="Local result ready")

    # Socket Check -> OFFLINE MODE
    add_edge("e13", "gw_socket_check", "off_mode_box", EDGE_OFFLINE, val="Offline (No Ping)")
    add_edge("e14", "off_mode_box", "api_response_builder", EDGE_OFFLINE, val="Direct Local Result")
    add_edge("e15", "api_response_builder", "ui_offline_badge", EDGE_OFFLINE, val="Sets is_offline=True")

    # Socket Check -> ONLINE MODE
    add_edge("e16", "gw_socket_check", "on_ai_check", EDGE_STD, val="Online (Ping OK)")
    add_edge("e17", "on_ai_check", "on_judge", EDGE_STD, val="Cloud Consensus")
    add_edge("e18", "on_judge", "on_reward", EDGE_REWARD, val="Correct Match")
    add_edge("e19", "on_judge", "on_punish", EDGE_PUNISH, val="Mismatch")

    # Punishment -> Memory and Weights Retraining
    add_edge("e20", "on_punish", "on_mem_store", EDGE_PUNISH, val="Save Vector")
    add_edge("e21", "on_mem_store", "ml_mem_check", EDGE_PUNISH, val="Update online_memory.pt", points=[(1090, 770), (620, 770), (620, 155)])
    add_edge("e22", "on_reward", "api_response_builder", EDGE_REWARD, val="Verified Result")
    add_edge("e23", "on_punish", "api_response_builder", EDGE_PUNISH, val="Corrected Result")

    # Response Builder -> UI Elements
    add_edge("e24", "api_response_builder", "ui_hud", EDGE_STD, val="JSON Payload")
    add_edge("e25", "api_response_builder", "ui_cam_toggle", EDGE_STD)
    add_edge("e26", "api_response_builder", "ui_multitarget", EDGE_STD)

    # Assets linkages
    add_edge("e27", "ast_mnet", "stg1_mobilenet", EDGE_STD, val="Loads weights")
    add_edge("e28", "ast_kb_embeds", "stg2_kb", EDGE_STD, val="512-D vectors")

    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ", level=0)
    return ET.tostring(root, encoding="utf-8", xml_declaration=True).decode("utf-8")


if __name__ == "__main__":
    xml_content = create_drawio_xml()
    out_path = Path("E:/PROJECTS/ASTRA-VISION/architecture.drawio")
    out_path.write_text(xml_content, encoding="utf-8")
    print(f"Successfully generated architecture.drawio at {out_path} ({len(xml_content)} bytes)")
