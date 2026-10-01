#!/usr/bin/env python3
"""Generates a clean, simple, plain-English draw.io diagram for ASTRA-VISION 5.0 with proper HTML rendering."""

import xml.etree.ElementTree as ET
from pathlib import Path


def create_simple_drawio_xml() -> str:
    root = ET.Element("mxfile", {
        "host": "app.diagrams.net",
        "modified": "2026-10-01T21:46:00.000Z",
        "agent": "ASTRA-VISION Architect",
        "version": "24.0.0",
        "type": "device",
    })

    diagram = ET.SubElement(root, "diagram", {
        "id": "astra-vision-simple-arch",
        "name": "ASTRA-VISION 5.0 Architecture",
    })

    model = ET.SubElement(diagram, "mxGraphModel", {
        "dx": "1400",
        "dy": "850",
        "grid": "1",
        "gridSize": "10",
        "guides": "1",
        "tooltips": "1",
        "connect": "1",
        "arrows": "1",
        "fold": "1",
        "page": "1",
        "pageScale": "1",
        "pageWidth": "1440",
        "pageHeight": "820",
        "background": "#0f172a",
        "math": "0",
        "shadow": "0",
    })

    root_cells = ET.SubElement(model, "root")
    ET.SubElement(root_cells, "mxCell", {"id": "0"})
    ET.SubElement(root_cells, "mxCell", {"id": "1", "parent": "0"})

    def add_node(id_str, val, style, x, y, w, h, parent="1"):
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

    # STYLES (Notice html=1 is required for HTML rendering)
    STYLE_TITLE = "text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;fontColor=#ffffff;fontSize=22;fontStyle=1;"
    STYLE_SUBTITLE = "text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;fontColor=#94a3b8;fontSize=13;fontStyle=0;"

    # Cards - html=1 enables rich bolding and linebreaks
    CARD_BLUE = "rounded=1;arcSize=10;whiteSpace=wrap;html=1;fillColor=#1e293b;strokeColor=#38bdf8;fontColor=#ffffff;fontSize=12;strokeWidth=2;align=left;spacingLeft=12;spacingRight=12;"
    CARD_PURPLE = "rounded=1;arcSize=10;whiteSpace=wrap;html=1;fillColor=#2e1065;strokeColor=#a855f7;fontColor=#ffffff;fontSize=12;strokeWidth=2;align=left;spacingLeft=12;spacingRight=12;"
    CARD_GREEN = "rounded=1;arcSize=10;whiteSpace=wrap;html=1;fillColor=#064e3b;strokeColor=#10b981;fontColor=#ffffff;fontSize=12;strokeWidth=2;align=left;spacingLeft=12;spacingRight=12;"
    CARD_AMBER = "rounded=1;arcSize=10;whiteSpace=wrap;html=1;fillColor=#451a03;strokeColor=#f59e0b;fontColor=#ffffff;fontSize=12;strokeWidth=2;align=left;spacingLeft=12;spacingRight=12;"
    CARD_PINK = "rounded=1;arcSize=10;whiteSpace=wrap;html=1;fillColor=#4c0519;strokeColor=#f43f5e;fontColor=#ffffff;fontSize=12;strokeWidth=2;align=left;spacingLeft=12;spacingRight=12;"
    CARD_DARK = "rounded=1;arcSize=10;whiteSpace=wrap;html=1;fillColor=#18181b;strokeColor=#64748b;fontColor=#ffffff;fontSize=12;strokeWidth=1.5;align=left;spacingLeft=12;spacingRight=12;"
    CARD_FUTURE = "rounded=1;arcSize=10;whiteSpace=wrap;html=1;fillColor=#1e293b;strokeColor=#64748b;fontColor=#f8fafc;fontSize=11;strokeWidth=1;align=left;spacingLeft=10;spacingRight=10;"

    CONTAINER_FUTURE = "swimlane;whiteSpace=wrap;html=1;rounded=1;arcSize=8;fillColor=#090d16;strokeColor=#64748b;fontColor=#e2e8f0;fontStyle=1;fontSize=13;dashed=1;strokeWidth=1.5;startSize=28;"

    # Connectors
    EDGE_MAIN = "edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;html=1;strokeColor=#38bdf8;strokeWidth=2;fontSize=11;fontColor=#bae6fd;"
    EDGE_OFFLINE = "edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;html=1;strokeColor=#f59e0b;strokeWidth=2.5;dashed=1;fontSize=11;fontColor=#fde68a;"
    EDGE_ONLINE = "edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;html=1;strokeColor=#f43f5e;strokeWidth=2;fontSize=11;fontColor=#fbcfe8;"
    EDGE_RETURN = "edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;html=1;strokeColor=#10b981;strokeWidth=2;fontSize=11;fontColor=#a7f3d0;"

    # =========================================================================
    # HEADER
    # =========================================================================
    add_node("hdr_title", "ASTRA-VISION 5.0 — SYSTEM ARCHITECTURE", STYLE_TITLE, 40, 15, 1360, 30)
    add_node("hdr_sub", "Simple End-to-End Pipeline: From Image Upload to Tactical Intel & Offline Mode", STYLE_SUBTITLE, 40, 45, 1360, 20)

    # =========================================================================
    # ROW 1: THE CORE 5-STEP PIPELINE (Left to Right)
    # NOTE: Use literal < and > so ElementTree escapes them ONCE into &lt; and &gt;
    # =========================================================================
    # 1. User & HUD
    node1_text = (
        "<b>1. USER & TACTICAL HUD</b><br><br>"
        "• Drag & drop photos (JPG, PNG)<br>"
        "• Shows Category & Confidence %<br>"
        "• Clickable Multi-Target tabs<br>"
        "• Batch upload up to 30 photos"
    )
    add_node("n1_hud", node1_text, CARD_BLUE, 40, 85, 230, 180)

    # 2. FastAPI Backend
    node2_text = (
        "<b>2. BACKEND API</b><br><br>"
        "• Built with <b>FastAPI</b> in Python<br>"
        "• High speed (under 1ms overhead)<br>"
        "• 10MB safety check & clean errors<br>"
        "• Rejects empty or broken files"
    )
    add_node("n2_api", node2_text, CARD_DARK, 310, 85, 220, 180)

    # 3. Processing (Multi-Object)
    node3_text = (
        "<b>3. PROCESSING</b><br><br>"
        "• <b>Multi-Object Engine</b><br>"
        "• Finds multiple vehicles/targets<br>"
        "• Draws clean bounding boxes<br>"
        "• Cuts high-res crops for each"
    )
    add_node("n3_proc", node3_text, CARD_DARK, 570, 85, 220, 180)

    # 4. AI / ML Models (Hybrid Ensemble)
    node4_text = (
        "<b>4. AI / ML MODELS</b><br><br>"
        "• <b>Input goes to local model FIRST</b><br>"
        "• <b>MobileNetV3</b> (18ms) + <b>CLIP</b> fusion<br>"
        "• <b>90.4% Test Accuracy</b> (0.9898 ROC)<br>"
        "• <b>Actual Calibrated Confidence %</b><br>"
        "  (NOT dummy score; tight margins penalized!)"
    )
    add_node("n4_ml", node4_text, CARD_PURPLE, 830, 85, 260, 185)

    # 5. Defense Knowledge Base
    node5_text = (
        "<b>5. DEFENSE DATABASE</b><br><br>"
        "• <b>74+ Military Platforms</b><br>"
        "• 512-D precomputed vectors<br>"
        "• Origin, year built & fleet count<br>"
        "• Speed, range & combat weapons"
    )
    add_node("n5_db", node5_text, CARD_GREEN, 1130, 85, 230, 185)

    # Connectors for Row 1
    add_edge("e1_2", "n1_hud", "n2_api", EDGE_MAIN, val="Upload")
    add_edge("e2_3", "n2_api", "n3_proc", EDGE_MAIN, val="Check & Send")
    add_edge("e3_4", "n3_proc", "n4_ml", EDGE_MAIN, val="Target Crops")
    add_edge("e4_5", "n4_ml", "n5_db", EDGE_MAIN, val="512-D Match")

    # =========================================================================
    # ROW 2: RESULT PROCESSING, OFFLINE MODE & SELF-LEARNING
    # =========================================================================
    # 6. Explainable AI (Grad-CAM)
    node6_text = (
        "<b>6. EXPLAINABLE AI (Grad-CAM)</b><br><br>"
        "• Shows a color heatmap directly on photo<br>"
        "• Proves AI looks at wings, rotors & armor<br>"
        "• Operator toggle switch in HUD"
    )
    add_node("n6_cam", node6_text, CARD_BLUE, 40, 310, 270, 180)

    # 7. Complete Dossier Result
    node7_text = (
        "<b>7. TACTICAL RESULT & REPORT</b><br><br>"
        "• Exact platform name (e.g. F-22 Raptor)<br>"
        "• Top-3 ranked categories with %<br>"
        "• <b>Actual Calibrated Confidence %</b> (never dummy!)<br>"
        "• Amber warning if confidence is under 60%<br>"
        "• Saved into local history gallery"
    )
    add_node("n7_res", node7_text, CARD_GREEN, 340, 310, 280, 180)

    # 8. Offline Mode (Latest V5 Feature)
    node8_text = (
        "<b>8. OFFLINE MODE (LATEST V5)</b><br><br>"
        "• Input tries local trained model FIRST<br>"
        "• Tries to reach API keys → <b>No internet detected!</b><br>"
        "• <b>Skips API keys entirely with zero lag</b><br>"
        "• Shows local prediction + <b>actual confidence %</b><br>"
        "• Displays glowing <b>'OFFLINE MODE'</b> badge"
    )
    add_node("n8_offline", node8_text, CARD_AMBER, 650, 310, 330, 180)

    # 9. Two-Stage Online Self-Learning (V5)
    node9_text = (
        "<b>9. TWO-STAGE SELF-LEARNING (V5)</b><br><br>"
        "• <b>1. Local model predicts FIRST</b><br>"
        "• <b>2. Verified by big-player API keys (Gemini/Groq)</b><br>"
        "• <b>If Correct?</b> → <font color='#10b981'>+10 Reward</font> & shows verified output<br>"
        "• <b>If Wrong?</b> → <font color='#f43f5e'>-50 Severe Penalty</font>, retrains weights &<br>"
        "  memorizes image so next time similar/same image is seen,<br>"
        "  it predicts it correctly from its previous mistake!"
    )
    add_node("n9_learn", node9_text, CARD_PINK, 1010, 310, 390, 180)

    # Connectors for Row 2
    add_edge("e5_6", "n4_ml", "n6_cam", EDGE_MAIN, val="Gradients", points=[(955, 280), (175, 280)])
    add_edge("e5_7", "n5_db", "n7_res", EDGE_MAIN, val="Specs", points=[(1240, 280), (485, 280)])
    add_edge("e7_hud", "n7_res", "n1_hud", EDGE_RETURN, val="Return to HUD", points=[(485, 505), (175, 505)])

    # Fork to Offline & Online
    add_edge("e_to_off", "n3_proc", "n8_offline", EDGE_OFFLINE, val="If No Internet", points=[(680, 280), (825, 280)])
    add_edge("e_to_on", "n5_db", "n9_learn", EDGE_ONLINE, val="If Online", points=[(1240, 280), (1215, 280)])
    add_edge("e_off_hud", "n8_offline", "n7_res", EDGE_OFFLINE, val="Local Result", points=[(660, 420), (620, 420)])

    # =========================================================================
    # ROW 3: FUTURE EXTENSIONS (ROADMAP)
    # =========================================================================
    add_node("c_future", "10. FUTURE ROADMAP (PLANNED IMPROVEMENTS)", CONTAINER_FUTURE, 40, 520, 1360, 260)

    # Card 1: Sound Prediction
    card1_text = (
        "<b>1. Sound-Based Prediction</b><br><br>"
        "• Predicts objects from sound<br>"
        "• Listens to jet engine roar, rotor<br>"
        "  frequencies & artillery blasts<br>"
        "• Works in fog & pitch darkness"
    )
    add_node("f1_sound", card1_text, CARD_FUTURE, 60, 565, 300, 185, parent="c_future")

    # Card 2: Live Camera Mode
    card2_text = (
        "<b>2. Live Camera Mode</b><br><br>"
        "• Like <b>Gemini Live</b><br>"
        "• Point camera at sky or vehicle<br>"
        "• Real-time live bounding boxes<br>"
        "• Instant tactical info overlay"
    )
    add_node("f2_live", card2_text, CARD_FUTURE, 380, 565, 300, 185, parent="c_future")

    # Card 3: Text Feature Search
    card3_text = (
        "<b>3. Text Feature Search</b><br><br>"
        "• Type observed physical features<br>"
        "• Example: <i>'delta wing, twin fins,<br>"
        "  dual engine, refueling probe'</i><br>"
        "• Model finds matching aircraft"
    )
    add_node("f3_text", card3_text, CARD_FUTURE, 700, 565, 310, 185, parent="c_future")

    # Card 4: Bangalore Flight News Scraper
    card4_text = (
        "<b>4. Contextual Flight & News Scraper</b><br><br>"
        "• User: <i>'I saw a fighter jet today over my<br>"
        "  house in Bangalore'</i><br>"
        "• Model scrapes news & flight radar to find<br>"
        "  local sorties (e.g. HAL Tejas / Su-30MKI)<br>"
        "• Displays which aircraft flew & their specs"
    )
    add_node("f4_osint", card4_text, CARD_FUTURE, 1030, 565, 350, 185, parent="c_future")

    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ", level=0)
    return ET.tostring(root, encoding="utf-8", xml_declaration=True).decode("utf-8")


if __name__ == "__main__":
    xml_content = create_simple_drawio_xml()
    out_path = Path("E:/PROJECTS/ASTRA-VISION/architecture.drawio")
    out_path.write_text(xml_content, encoding="utf-8")
    print(f"Successfully fixed architecture.drawio ({len(xml_content)} bytes)")
