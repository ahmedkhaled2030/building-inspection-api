"""
Building Inspection System - Backend API
=========================================
FastAPI server that processes thermal images using a YOLOv8 model
to detect moisture and water leaks, generates annotated images
and professional PDF reports.
"""

import os
import uuid
import base64
import tempfile
from datetime import datetime
from pathlib import Path
from typing import List

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from ultralytics import YOLO
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch, mm
from reportlab.lib.colors import HexColor, white, black
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image as RLImage,
    Table, TableStyle, PageBreak,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT

# ---------------------------------------------------------------------------
# App & Model Initialization
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Building Inspection API",
    description="Detects moisture and water leaks in thermal images",
    version="1.0.0",
)

# CORS – allow the React dev server and production frontend
_cors_origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]
# Add production frontend URL from env var (e.g. https://your-app.vercel.app)
_extra_origin = os.environ.get("FRONTEND_URL")
if _extra_origin:
    _cors_origins.append(_extra_origin.rstrip("/"))

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the YOLO model once at startup
MODEL_PATH = Path(__file__).parent / "best.pt"
model = YOLO(str(MODEL_PATH))

# Directory to persist generated PDF reports
REPORTS_DIR = Path(__file__).parent / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# NMS / Inference tuning constants
CONF_THRESHOLD = 0.45   # Minimum confidence to keep a detection
IOU_THRESHOLD  = 0.3    # IoU threshold for NMS – lower = more aggressive merging


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _severity(confidence: float) -> str:
    """Map confidence % to a human-readable severity label."""
    if confidence >= 80:
        return "High"
    if confidence >= 50:
        return "Medium"
    return "Low"


def _severity_color(confidence: float) -> str:
    """Hex colour string for the severity badge in the PDF."""
    if confidence >= 80:
        return "#dc2626"   # red
    if confidence >= 50:
        return "#f59e0b"   # amber
    return "#6b7280"       # grey


# ---------------------------------------------------------------------------
# Helper: validate uploaded file
# ---------------------------------------------------------------------------
def _validate_image(filename: str) -> None:
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"File '{filename}' is not a supported image format. "
                   f"Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )


# ---------------------------------------------------------------------------
# Helper: run inference on a single image and return annotated image + stats
# ---------------------------------------------------------------------------
def _process_image(img_bytes: bytes, filename: str):
    """Run YOLO inference and draw bounding boxes.

    Returns:
        annotated_b64: base64-encoded annotated JPEG
        detections: list of dicts with detection info
        annotated_bytes: raw JPEG bytes (for PDF embedding)
    """
    # Decode the raw bytes into an OpenCV image
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail=f"Cannot decode image '{filename}'.")

    # Run YOLOv8 inference with strict NMS to eliminate overlapping boxes
    results = model.predict(
        img,
        verbose=False,
        conf=CONF_THRESHOLD,
        iou=IOU_THRESHOLD,
        agnostic_nms=True,      # merge overlapping boxes regardless of class
        max_det=50,             # safety cap
    )
    result = results[0]

    # Draw clean bounding boxes via Ultralytics built-in renderer
    annotated = result.plot(line_width=2, font_size=12)

    # Collect per-detection stats
    detections: List[dict] = []
    boxes = result.boxes
    if boxes is not None and len(boxes) > 0:
        for box in boxes:
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            cls_name = result.names.get(cls_id, f"class_{cls_id}")
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf_pct = round(conf * 100, 1)
            detections.append({
                "class": cls_name,
                "confidence": conf_pct,
                "severity": _severity(conf_pct),
                "bbox": [round(v, 1) for v in [x1, y1, x2, y2]],
            })

    # Encode annotated image to JPEG bytes
    _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 92])
    annotated_bytes = buf.tobytes()
    annotated_b64 = base64.b64encode(annotated_bytes).decode("utf-8")

    return annotated_b64, detections, annotated_bytes


# ---------------------------------------------------------------------------
# PDF Report – Premium multi-page layout
# ---------------------------------------------------------------------------
# Colour palette
_NAVY       = HexColor("#0f2a4a")
_DARK_BLUE  = HexColor("#1e3a5f")
_MID_BLUE   = HexColor("#2c5282")
_LIGHT_BLUE = HexColor("#e8eef5")
_ACCENT     = HexColor("#2b8cc4")
_GREY_50    = HexColor("#f9fafb")
_GREY_100   = HexColor("#f3f4f6")
_GREY_200   = HexColor("#e5e7eb")
_GREY_500   = HexColor("#6b7280")
_GREY_700   = HexColor("#374151")
_TXT_DARK   = HexColor("#1f2937")

PAGE_W, PAGE_H = A4
MARGIN = 20 * mm


def _build_styles() -> dict:
    """Return a dict of named ParagraphStyles for the report."""
    base = getSampleStyleSheet()
    return {
        "cover_title": ParagraphStyle(
            "CoverTitle", parent=base["Title"],
            fontSize=30, leading=36, textColor=white,
            alignment=TA_CENTER, spaceAfter=10,
            fontName="Helvetica-Bold",
        ),
        "cover_subtitle": ParagraphStyle(
            "CoverSub", parent=base["Normal"],
            fontSize=14, leading=18, textColor=HexColor("#cbd5e1"),
            alignment=TA_CENTER, spaceAfter=4,
        ),
        "cover_date": ParagraphStyle(
            "CoverDate", parent=base["Normal"],
            fontSize=12, textColor=HexColor("#94a3b8"),
            alignment=TA_CENTER, spaceBefore=20,
        ),
        "section_title": ParagraphStyle(
            "SectionTitle", parent=base["Heading1"],
            fontSize=18, leading=22, textColor=_DARK_BLUE,
            fontName="Helvetica-Bold", spaceBefore=6, spaceAfter=10,
        ),
        "heading": ParagraphStyle(
            "Heading", parent=base["Heading2"],
            fontSize=13, leading=16, textColor=_DARK_BLUE,
            fontName="Helvetica-Bold", spaceBefore=12, spaceAfter=6,
        ),
        "body": ParagraphStyle(
            "Body", parent=base["BodyText"],
            fontSize=10, leading=14, textColor=_TXT_DARK,
        ),
        "body_center": ParagraphStyle(
            "BodyCenter", parent=base["BodyText"],
            fontSize=10, leading=14, textColor=_TXT_DARK,
            alignment=TA_CENTER,
        ),
        "small_grey": ParagraphStyle(
            "SmallGrey", parent=base["Normal"],
            fontSize=8, textColor=_GREY_500, alignment=TA_CENTER,
        ),
    }


def _hr(width: float, color=_GREY_200, thickness=0.75):
    """Return a thin horizontal rule as a 1-cell table."""
    t = Table([[""]], colWidths=[width])
    t.setStyle(TableStyle([
        ("LINEABOVE", (0, 0), (-1, 0), thickness, color),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
    ]))
    return t


def _generate_pdf(image_results: List[dict]) -> str:
    """Build a premium, multi-page PDF inspection report."""

    report_id = uuid.uuid4().hex[:8].upper()
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    pdf_filename = f"inspection_report_{timestamp}_{report_id}.pdf"
    pdf_path = REPORTS_DIR / pdf_filename

    content_width = PAGE_W - 2 * MARGIN

    doc = SimpleDocTemplate(
        str(pdf_path), pagesize=A4,
        topMargin=MARGIN, bottomMargin=MARGIN,
        leftMargin=MARGIN, rightMargin=MARGIN,
    )

    S = _build_styles()
    elements: list = []
    tmp_files: list = []          # temp image files to clean up later

    # ================================================================
    # PAGE 1 – COVER PAGE
    # ================================================================
    cover_content = [
        [Paragraph("<br/><br/><br/><br/><br/>", S["body"])],
        [Paragraph("THERMAL BUILDING<br/>INSPECTION REPORT", S["cover_title"])],
        [Paragraph("<br/>", S["body"])],
        [Paragraph("Infrared Thermographic Survey<br/>Moisture &amp; Water Leak Assessment", S["cover_subtitle"])],
        [Paragraph("<br/><br/>", S["body"])],
        [Paragraph(f'<font color="#ffffff">━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━</font>', S["body_center"])],
        [Paragraph("<br/>", S["body"])],
        [Paragraph(f"Report Reference: &nbsp; TBI-{report_id}", S["cover_subtitle"])],
        [Paragraph(f"Date of Inspection: &nbsp; {now.strftime('%B %d, %Y')}", S["cover_date"])],
        [Paragraph(f"Issued: &nbsp; {now.strftime('%B %d, %Y  —  %H:%M')}", S["cover_date"])],
        [Paragraph("<br/><br/>", S["body"])],
        [Paragraph("Prepared for: ____________________________________", S["cover_date"])],
        [Paragraph("<br/>", S["body"])],
        [Paragraph("Site Address: ____________________________________", S["cover_date"])],
        [Paragraph("<br/><br/><br/><br/><br/><br/>", S["body"])],
        [Paragraph("CONFIDENTIAL", S["cover_subtitle"])],
        [Paragraph("ThermaVision Diagnostics  •  Building Envelope Specialists", S["small_grey"])],
    ]
    cover_table = Table(cover_content, colWidths=[content_width])
    cover_table.setStyle(TableStyle([
        ("BACKGROUND",  (0, 0), (-1, -1), _NAVY),
        ("TOPPADDING",  (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
        ("LEFTPADDING", (0, 0), (-1, -1), 20),
        ("RIGHTPADDING",(0, 0), (-1, -1), 20),
        ("VALIGN",      (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN",       (0, 0), (-1, -1), "CENTER"),
        ("LINEABOVE",   (0, 0), (-1, 0), 2, _NAVY),
        ("LINEBELOW",   (0, -1), (-1, -1), 2, _NAVY),
    ]))
    elements.append(cover_table)
    elements.append(PageBreak())

    # ================================================================
    # PAGE 2 – EXECUTIVE SUMMARY
    # ================================================================
    total_images     = len(image_results)
    total_detections = sum(len(r["detections"]) for r in image_results)
    all_confs        = [d["confidence"] for r in image_results for d in r["detections"]]
    avg_conf         = round(sum(all_confs) / len(all_confs), 1) if all_confs else 0.0
    high_count       = sum(1 for c in all_confs if c >= 80)
    medium_count     = sum(1 for c in all_confs if 50 <= c < 80)
    low_count        = sum(1 for c in all_confs if c < 50)

    elements.append(Paragraph("Executive Summary", S["section_title"]))
    elements.append(_hr(content_width, _DARK_BLUE, 1.5))
    elements.append(Spacer(1, 10))

    elements.append(Paragraph(
        "This report presents the findings of a comprehensive infrared thermographic survey "
        "conducted on the subject property. The inspection was performed to evaluate the "
        "thermal performance of the building envelope and to identify areas exhibiting "
        "abnormal thermal patterns indicative of moisture ingress, water infiltration, "
        "or compromised insulation integrity.",
        S["body"],
    ))
    elements.append(Spacer(1, 6))
    elements.append(Paragraph(
        "All thermal images were processed through our proprietary diagnostic engine, which "
        "applies advanced pattern recognition to isolate regions of concern with high precision. "
        "Each finding is classified by severity to assist in prioritising remediation efforts.",
        S["body"],
    ))
    elements.append(Spacer(1, 14))

    # Summary KPI cards as a table
    kpi_data = [
        ["Images\nInspected", "Anomalies\nIdentified", "Avg\nCertainty",
         "Critical\nFindings", "Moderate\nFindings", "Minor\nFindings"],
        [str(total_images), str(total_detections), f"{avg_conf}%",
         str(high_count), str(medium_count), str(low_count)],
    ]
    col_w = content_width / 6
    kpi_table = Table(kpi_data, colWidths=[col_w] * 6, rowHeights=[32, 36])
    kpi_table.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1, 0), _DARK_BLUE),
        ("TEXTCOLOR",    (0, 0), (-1, 0), white),
        ("FONTNAME",     (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",     (0, 0), (-1, 0), 8),
        ("ALIGN",        (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
        ("BACKGROUND",   (0, 1), (-1, 1), _LIGHT_BLUE),
        ("FONTNAME",     (0, 1), (-1, 1), "Helvetica-Bold"),
        ("FONTSIZE",     (0, 1), (-1, 1), 16),
        ("TEXTCOLOR",    (0, 1), (-1, 1), _DARK_BLUE),
        ("GRID",         (0, 0), (-1, -1), 0.5, white),
        ("TOPPADDING",   (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 6),
    ]))
    elements.append(kpi_table)
    elements.append(Spacer(1, 18))

    # Inspection Scope & Methodology
    elements.append(Paragraph("Inspection Scope &amp; Methodology", S["heading"]))
    elements.append(Paragraph(
        "The thermal survey was carried out using calibrated infrared thermographic imaging "
        "equipment in accordance with industry-standard building envelope assessment practices. "
        "Captured thermal images were analysed through our proprietary diagnostic platform, "
        "which applies multi-stage anomaly detection and spatial filtering to ensure each "
        "area of concern is reported precisely without duplication.",
        S["body"],
    ))
    elements.append(Spacer(1, 6))
    elements.append(Paragraph(
        "Each identified anomaly is assigned a severity classification based on diagnostic certainty: "
        "<b>Critical</b> (≥ 80% certainty) — requires immediate remediation; "
        "<b>Moderate</b> (50–79% certainty) — scheduled follow-up recommended; "
        "<b>Minor</b> (&lt; 50% certainty) — monitor during next routine inspection.",
        S["body"],
    ))
    elements.append(Spacer(1, 10))

    # Overall risk assessment
    if high_count > 0:
        risk_text = (
            f'<font color="#dc2626"><b>URGENT — IMMEDIATE ACTION REQUIRED</b></font><br/>'
            f'{high_count} critical anomal{"y" if high_count == 1 else "ies"} identified '
            f'with high diagnostic certainty. Active moisture ingress is highly probable. '
            f'Immediate physical investigation and remediation planning is strongly recommended '
            f'to prevent further structural deterioration.'
        )
    elif medium_count > 0:
        risk_text = (
            f'<font color="#f59e0b"><b>MODERATE RISK — FOLLOW-UP RECOMMENDED</b></font><br/>'
            f'{medium_count} moderate anomal{"y" if medium_count == 1 else "ies"} identified. '
            f'Thermal signatures suggest potential moisture presence. A targeted follow-up '
            f'inspection should be scheduled within 30 days to assess progression.'
        )
    elif total_detections > 0:
        risk_text = (
            f'<font color="#16a34a"><b>LOW RISK — ROUTINE MONITORING</b></font><br/>'
            f'Only minor thermal irregularities were detected. These may be attributed to '
            f'normal thermal bridging or transient environmental conditions. Continue routine '
            f'monitoring during scheduled maintenance cycles.'
        )
    else:
        risk_text = (
            f'<font color="#16a34a"><b>SATISFACTORY — NO ANOMALIES DETECTED</b></font><br/>'
            f'The inspected areas exhibit normal thermal performance. No indications of '
            f'moisture ingress or insulation deficiency were identified at the time of survey.'
        )

    elements.append(Paragraph("Overall Risk Assessment", S["heading"]))
    elements.append(Paragraph(risk_text, S["body"]))

    # Recommendations summary
    elements.append(Spacer(1, 10))
    elements.append(Paragraph("Recommendations", S["heading"]))
    if high_count > 0:
        elements.append(Paragraph(
            "1. &nbsp; Commission an invasive moisture survey at all critical locations identified in this report.<br/>"
            "2. &nbsp; Engage a waterproofing specialist to assess the building envelope at affected areas.<br/>"
            "3. &nbsp; Document all remediation work performed and schedule a post-repair thermographic verification.<br/>"
            "4. &nbsp; Review building maintenance records for prior water damage at identified zones.",
            S["body"],
        ))
    elif medium_count > 0:
        elements.append(Paragraph(
            "1. &nbsp; Schedule a follow-up thermographic inspection within 30 days to monitor progression.<br/>"
            "2. &nbsp; Visually inspect the exterior envelope at flagged locations for visible defects.<br/>"
            "3. &nbsp; Consider moisture meter readings at moderate-risk zones for confirmation.",
            S["body"],
        ))
    elif total_detections > 0:
        elements.append(Paragraph(
            "1. &nbsp; No immediate action required.<br/>"
            "2. &nbsp; Include identified areas in the next routine maintenance inspection cycle.<br/>"
            "3. &nbsp; Re-inspect if occupants report dampness or condensation symptoms.",
            S["body"],
        ))
    else:
        elements.append(Paragraph(
            "1. &nbsp; No remediation action required at this time.<br/>"
            "2. &nbsp; Maintain regular inspection intervals as per building maintenance schedule.",
            S["body"],
        ))

    elements.append(PageBreak())

    # ================================================================
    # PAGES 3+ – DETAILED IMAGE ANALYSIS (one page per image)
    # ================================================================
    elements.append(Paragraph("Detailed Findings", S["section_title"]))
    elements.append(_hr(content_width, _DARK_BLUE, 1.5))
    elements.append(Spacer(1, 6))

    for idx, result in enumerate(image_results, start=1):
        if idx > 1:
            elements.append(PageBreak())

        fname = result["filename"]
        dets  = result["detections"]

        elements.append(Paragraph(
            f'Thermogram {idx} of {total_images}: &nbsp; <font color="#2c5282">{fname}</font>',
            S["heading"],
        ))
        elements.append(Spacer(1, 4))

        # Write annotated image to temp file for ReportLab
        tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        tmp.write(result["annotated_bytes"])
        tmp.close()
        tmp_files.append(tmp.name)

        img_obj = RLImage(tmp.name, width=content_width, height=4.2 * inch, kind="proportional")
        elements.append(img_obj)
        elements.append(Spacer(1, 10))

        if dets:
            # Detection details table with severity
            header = [["Finding", "Type", "Certainty", "Severity", "Recommended Action"]]
            rows = []
            for di, d in enumerate(dets, 1):
                sev = d["severity"]
                sev_color = _severity_color(d["confidence"])
                sev_label = (
                    "Critical" if sev == "High" else
                    "Moderate" if sev == "Medium" else
                    "Minor"
                )
                obs = (
                    "Immediate invasive investigation and remediation required."
                    if sev == "High" else
                    "Schedule targeted follow-up inspection within 30 days."
                    if sev == "Medium" else
                    "Monitor during next routine maintenance cycle."
                )
                rows.append([
                    f"MF-{idx}.{di}",
                    d["class"].replace("_", " ").title(),
                    f'{d["confidence"]}%',
                    Paragraph(f'<font color="{sev_color}"><b>{sev_label}</b></font>', S["body_center"]),
                    Paragraph(obs, S["body"]),
                ])

            det_table = Table(
                header + rows,
                colWidths=[
                    0.7 * inch,    # Finding ref
                    0.95 * inch,   # Type
                    0.75 * inch,   # Certainty
                    0.75 * inch,   # Severity
                    content_width - 3.15 * inch,  # Recommended Action
                ],
            )
            det_table.setStyle(TableStyle([
                ("BACKGROUND",    (0, 0), (-1, 0), _DARK_BLUE),
                ("TEXTCOLOR",     (0, 0), (-1, 0), white),
                ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE",      (0, 0), (-1, 0), 9),
                ("ALIGN",         (0, 0), (-1, 0), "CENTER"),
                ("FONTSIZE",      (0, 1), (-1, -1), 9),
                ("FONTNAME",      (0, 1), (-1, -1), "Helvetica"),
                ("ALIGN",         (0, 1), (2, -1), "CENTER"),
                ("ALIGN",         (3, 1), (3, -1), "CENTER"),
                ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [white, _GREY_50]),
                ("GRID",          (0, 0), (-1, -1), 0.4, _GREY_200),
                ("LINEBELOW",     (0, 0), (-1, 0), 1.2, _DARK_BLUE),
                ("TOPPADDING",    (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("LEFTPADDING",   (0, 0), (-1, -1), 6),
                ("RIGHTPADDING",  (0, 0), (-1, -1), 6),
            ]))
            elements.append(det_table)

            # Quick summary line
            elements.append(Spacer(1, 6))
            high_n = sum(1 for d in dets if d["severity"] == "High")
            med_n  = sum(1 for d in dets if d["severity"] == "Medium")
            low_n  = sum(1 for d in dets if d["severity"] == "Low")
            parts = []
            if high_n: parts.append(f'<font color="#dc2626"><b>{high_n} Critical</b></font>')
            if med_n:  parts.append(f'<font color="#f59e0b"><b>{med_n} Moderate</b></font>')
            if low_n:  parts.append(f'<font color="#6b7280"><b>{low_n} Minor</b></font>')
            elements.append(Paragraph(
                f"Thermogram summary: &nbsp; {' &nbsp;|&nbsp; '.join(parts)} "
                f"&nbsp; — &nbsp; {len(dets)} anomal{'y' if len(dets) == 1 else 'ies'} identified",
                S["body"],
            ))
        else:
            elements.append(Spacer(1, 4))
            elements.append(Paragraph(
                '<font color="#16a34a"><b>No thermal anomalies identified — area exhibits normal thermal performance.</b></font>',
                S["body"],
            ))

    # ================================================================
    # FINAL PAGE – Terms & Conditions
    # ================================================================
    elements.append(PageBreak())
    elements.append(Paragraph("Terms &amp; Conditions", S["section_title"]))
    elements.append(_hr(content_width, _DARK_BLUE, 1.5))
    elements.append(Spacer(1, 8))
    elements.append(Paragraph(
        "This report has been prepared by ThermaVision Diagnostics for the exclusive use of "
        "the client named on the cover page. The thermographic inspection and subsequent analysis "
        "were performed in accordance with accepted building science practices and are based on "
        "conditions observed at the time of survey.",
        S["body"],
    ))
    elements.append(Spacer(1, 6))
    elements.append(Paragraph(
        "Infrared thermography is a non-invasive, non-destructive assessment method. While it provides "
        "valuable diagnostic information, it does not replace invasive testing. The findings and "
        "recommendations in this report should be used as a guide for further investigation where indicated. "
        "ThermaVision Diagnostics accepts no liability for conditions not visible or detectable through "
        "infrared imaging at the time of inspection.",
        S["body"],
    ))
    elements.append(Spacer(1, 6))
    elements.append(Paragraph(
        "This report remains the intellectual property of ThermaVision Diagnostics and may not be "
        "reproduced, distributed, or disclosed to third parties without prior written consent, "
        "except in its entirety for the purpose of obtaining remediation quotations.",
        S["body"],
    ))
    elements.append(Spacer(1, 24))
    elements.append(_hr(content_width, _GREY_200, 0.5))
    elements.append(Spacer(1, 8))
    elements.append(Paragraph(
        f"Report Reference: TBI-{report_id} &nbsp;&nbsp;|&nbsp;&nbsp; "
        f"Issued: {now.strftime('%B %d, %Y at %H:%M')}",
        S["small_grey"],
    ))
    elements.append(Spacer(1, 4))
    elements.append(Paragraph(
        "© ThermaVision Diagnostics — All Rights Reserved",
        S["small_grey"],
    ))

    # ================================================================
    # BUILD & CLEANUP
    # ================================================================
    doc.build(elements)

    for tmp_path in tmp_files:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    return str(pdf_path)


# ---------------------------------------------------------------------------
# Main detection endpoint
# ---------------------------------------------------------------------------
@app.post("/api/detect")
async def detect(files: List[UploadFile] = File(...)):
    """Accept one or more thermal images, run YOLO inference, and return
    annotated images (base64), detection stats, and a PDF report."""

    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    # Validate all files first
    for f in files:
        _validate_image(f.filename or "unknown")

    image_results: List[dict] = []
    response_images: list = []

    for f in files:
        img_bytes = await f.read()
        annotated_b64, detections, annotated_bytes = _process_image(img_bytes, f.filename or "unknown")

        image_results.append({
            "filename": f.filename,
            "detections": detections,
            "annotated_bytes": annotated_bytes,
        })
        response_images.append({
            "filename": f.filename,
            "image_base64": annotated_b64,
            "detections": detections,
        })

    # Generate the PDF report
    pdf_path = _generate_pdf(image_results)
    pdf_filename = Path(pdf_path).name

    # Encode the PDF as base64 so the frontend can offer a download
    with open(pdf_path, "rb") as pf:
        pdf_b64 = base64.b64encode(pf.read()).decode("utf-8")

    return {
        "success": True,
        "total_images": len(response_images),
        "images": response_images,
        "pdf_base64": pdf_b64,
        "pdf_filename": pdf_filename,
    }


# ---------------------------------------------------------------------------
# Optional: direct PDF download endpoint
# ---------------------------------------------------------------------------
@app.get("/api/reports/{filename}")
async def download_report(filename: str):
    """Serve a previously generated PDF report for download."""
    # Sanitize filename to prevent path traversal
    safe_name = Path(filename).name
    file_path = REPORTS_DIR / safe_name
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Report not found.")
    return FileResponse(str(file_path), media_type="application/pdf", filename=safe_name)
