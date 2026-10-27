"""
PDF Report Generator for Brain Tumor Detection System.
Generates a clean, professional, single-page medical AI screening report
using ReportLab with patient details, scan image, prediction result, and disclaimer.
"""

import io
from typing import Union, Any
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image as PDFImage,
    Table,
    TableStyle,
    KeepTogether,
)


def generate_pdf_report(
    patient_name: str,
    age: Any,
    city: str,
    mobile_number: str,
    scan_datetime: str,
    mri_image: Union[PILImage.Image, bytes],
    prediction_result: str,
    confidence: float,
    gradcam_overlay: Union[PILImage.Image, None] = None,
) -> bytes:
    """
    Generate a standardized, publication-grade PDF report for the Brain Tumor Detection System.

    Args:
        patient_name: Name of the patient.
        age: Age of the patient.
        city: City/location of the patient.
        mobile_number: Contact number.
        scan_datetime: Timestamp of scan analysis.
        mri_image: PIL Image instance or raw bytes of the MRI scan.
        prediction_result: 'Tumor Detected' or 'No Tumor Detected'.
        confidence: Prediction confidence score as a percentage (0.0 to 100.0).

    Returns:
        bytes: Raw PDF file bytes.
    """
    pdf_buffer = io.BytesIO()

    # Document setup with 36pt (0.5 inch) margins for a balanced single-page layout
    doc = SimpleDocTemplate(
        pdf_buffer,
        pagesize=A4,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36,
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        name="DocTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        alignment=TA_LEFT,
        textColor=colors.HexColor("#0f172a"),
    )

    sub_title_style = ParagraphStyle(
        name="DocSubTitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=13,
        alignment=TA_LEFT,
        textColor=colors.HexColor("#0284c7"),
    )

    header_meta_style = ParagraphStyle(
        name="HeaderMeta",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=12,
        alignment=TA_RIGHT,
        textColor=colors.HexColor("#475569"),
    )

    section_heading = ParagraphStyle(
        name="SectionHeading",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=15,
        textColor=colors.HexColor("#0f172a"),
        spaceBefore=8,
        spaceAfter=4,
    )

    cell_label = ParagraphStyle(
        name="CellLabel",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#334155"),
    )

    cell_val = ParagraphStyle(
        name="CellVal",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#0f172a"),
    )

    disclaimer_text_style = ParagraphStyle(
        name="DisclaimerText",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#78350f"),
        alignment=TA_CENTER,
    )

    story = []

    # 1. HEADER SECTION
    header_data = [
        [
            Paragraph("<b>Brain Tumor Detection System</b>", title_style),
            Paragraph(f"<b>Scan Date & Time:</b><br/>{scan_datetime}", header_meta_style),
        ],
        [
            Paragraph("Deep Learning MRI Diagnostic Screening Report", sub_title_style),
            Paragraph("<font color='#0284c7'><b>Confidential Medical AI Report</b></font>", header_meta_style),
        ],
    ]
    header_table = Table(header_data, colWidths=[4.6 * inch, 2.7 * inch])
    header_table.setStyle(
        TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 0),
            ("TOPPADDING", (0, 0), (-1, -1), 1),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
            ("LINEBELOW", (0, 1), (-1, 1), 2, colors.HexColor("#0284c7")),
        ])
    )
    story.append(header_table)
    story.append(Spacer(1, 10))

    # 2. PATIENT DETAILS
    story.append(Paragraph("Patient Information", section_heading))
    patient_table_data = [
        [
            Paragraph("Patient Name", cell_label),
            Paragraph(f"<b>{patient_name}</b>", cell_val),
            Paragraph("Age", cell_label),
            Paragraph(f"{age} Years", cell_val),
        ],
        [
            Paragraph("City", cell_label),
            Paragraph(f"{city}", cell_val),
            Paragraph("Mobile Number", cell_label),
            Paragraph(f"{mobile_number}", cell_val),
        ],
        [
            Paragraph("Scan Date & Time", cell_label),
            Paragraph(f"{scan_datetime}", cell_val),
            Paragraph("Exam Type", cell_label),
            Paragraph("Brain MRI (Axial)", cell_val),
        ],
    ]
    patient_table = Table(
        patient_table_data,
        colWidths=[1.5 * inch, 2.15 * inch, 1.5 * inch, 2.15 * inch],
    )
    patient_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ])
    )
    story.append(patient_table)
    story.append(Spacer(1, 12))

    # 3. DETECTION RESULTS
    story.append(Paragraph("AI Diagnostic Result", section_heading))

    is_tumor = "tumor detected" in prediction_result.lower() and "no tumor" not in prediction_result.lower()
    if is_tumor:
        result_bg = colors.HexColor("#fee2e2")
        result_text_color = "#b91c1c"
        result_label = "Tumor Detected"
    else:
        result_bg = colors.HexColor("#dcfce7")
        result_text_color = "#15803d"
        result_label = "No Tumor Detected"

    pred_cell_style = ParagraphStyle(
        name="PredCell",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=16,
        textColor=colors.HexColor(result_text_color),
    )

    conf_cell_style = ParagraphStyle(
        name="ConfCell",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=15,
        textColor=colors.HexColor("#0f172a"),
    )

    results_data = [
        [
            Paragraph("Predicted Result", cell_label),
            Paragraph(f"<b>{result_label}</b>", pred_cell_style),
            Paragraph("Model Confidence", cell_label),
            Paragraph(f"<b>{confidence:.2f}%</b>", conf_cell_style),
        ],
    ]
    results_table = Table(
        results_data,
        colWidths=[1.5 * inch, 2.15 * inch, 1.5 * inch, 2.15 * inch],
    )
    results_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
            ("BACKGROUND", (1, 0), (1, 0), result_bg),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ("TOPPADDING", (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ])
    )
    story.append(results_table)
    story.append(Spacer(1, 12))

    # 4. UPLOADED MRI IMAGE
    story.append(Paragraph("Uploaded MRI Scan", section_heading))

    # Prepare Image buffer
    if isinstance(mri_image, bytes):
        pil_img = PILImage.open(io.BytesIO(mri_image)).convert("RGB")
    else:
        pil_img = mri_image.convert("RGB")

    img_buf = io.BytesIO()
    pil_img.save(img_buf, format="PNG")
    img_buf.seek(0)

    # Scale image to fit neatly on page
    img_size = 2.4 * inch
    pdf_img = PDFImage(img_buf, width=img_size, height=img_size)
    pdf_img.hAlign = "CENTER"

    img_caption = ParagraphStyle(
        name="ImgCaption",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=8,
        leading=11,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#64748b"),
    )

    image_table = Table(
        [
            [pdf_img],
            [Paragraph("Uploaded Brain MRI Scan (Axial View)", img_caption)],
        ],
        colWidths=[7.3 * inch],
    )
    image_table.setStyle(
        TableStyle([
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ])
    )
    story.append(image_table)
    story.append(Spacer(1, 12))

    # 5. GRAD-CAM OVERLAY (optional)
    if gradcam_overlay is not None:
        story.append(Paragraph("Grad-CAM Explainability Heatmap", section_heading))

        gradcam_buf = io.BytesIO()
        gradcam_overlay.convert("RGB").save(gradcam_buf, format="PNG")
        gradcam_buf.seek(0)

        gradcam_pdf_img = PDFImage(gradcam_buf, width=img_size, height=img_size)
        gradcam_pdf_img.hAlign = "CENTER"

        gradcam_table = Table(
            [
                [gradcam_pdf_img],
                [Paragraph("MRI + Grad-CAM Overlay (Regions most influencing the prediction)", img_caption)],
            ],
            colWidths=[7.3 * inch],
        )
        gradcam_table.setStyle(
            TableStyle([
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#7c3aed")),
            ])
        )
        story.append(gradcam_table)
        story.append(Spacer(1, 12))

    # 6. MANDATORY DISCLAIMER
    disclaimer_text = (
        "<b>Medical Disclaimer:</b> "
        "This report is generated by an AI model for educational purposes only. "
        "It is not a medical diagnosis. Please consult a qualified medical professional."
    )
    disclaimer_box = Table(
        [[Paragraph(disclaimer_text, disclaimer_text_style)]],
        colWidths=[7.3 * inch],
    )
    disclaimer_box.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fefce8")),
            ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#ca8a04")),
            ("TOPPADDING", (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ("LEFTPADDING", (0, 0), (-1, -1), 12),
            ("RIGHTPADDING", (0, 0), (-1, -1), 12),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ])
    )
    story.append(KeepTogether([disclaimer_box]))

    # Build the document
    doc.build(story)
    pdf_buffer.seek(0)
    return pdf_buffer.getvalue()
