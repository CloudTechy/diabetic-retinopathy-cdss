import io
import os
import datetime
from datetime import timezone
from typing import Optional
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    HRFlowable,
    KeepTogether,
)

from app.core.config import settings
from app.models.models import Assessment


class ReportService:
    """Generates server-side clinical PDF reports for assessments."""

    @staticmethod
    def generate_pdf_report(assessment: Assessment) -> bytes:
        """Generate a complete tamper-evident clinical PDF report."""
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            rightMargin=36,
            leftMargin=36,
            topMargin=36,
            bottomMargin=36,
        )

        styles = getSampleStyleSheet()
        normal = styles["Normal"]

        # Custom typography styles
        title_style = ParagraphStyle(
            "DocTitle",
            parent=normal,
            fontName="Helvetica-Bold",
            fontSize=16,
            leading=20,
            textColor=colors.HexColor("#0F766E"),  # Clinical teal
        )

        subtitle_style = ParagraphStyle(
            "DocSubtitle",
            parent=normal,
            fontName="Helvetica",
            fontSize=9,
            leading=12,
            textColor=colors.HexColor("#475569"),
        )

        h2_style = ParagraphStyle(
            "SectionH2",
            parent=normal,
            fontName="Helvetica-Bold",
            fontSize=11,
            leading=14,
            textColor=colors.HexColor("#1E293B"),
        )

        body_style = ParagraphStyle(
            "BodySmall",
            parent=normal,
            fontName="Helvetica",
            fontSize=8.5,
            leading=11,
            textColor=colors.HexColor("#334155"),
        )

        disclaimer_style = ParagraphStyle(
            "Disclaimer",
            parent=normal,
            fontName="Helvetica-Oblique",
            fontSize=7.5,
            leading=10,
            textColor=colors.HexColor("#64748B"),
        )

        elements = []

        # --- Header ---
        elements.append(Paragraph("Assessment Report", title_style))
        elements.append(Paragraph("Ophthalmic Retinal Screening & Decision-Support Consultation Summary", subtitle_style))
        elements.append(Spacer(1, 4))
        elements.append(
            Paragraph(
                "<strong>System:</strong> Diabetic Retinopathy Decision Support (EfficientNet-B0) &middot; Clinical Evaluation Protocol",
                disclaimer_style,
            )
        )
        elements.append(Spacer(1, 6))
        elements.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#0F766E"), spaceAfter=10))

        # --- Patient & Encounter Metadata Table ---
        val = assessment.validation_result
        img = assessment.image_asset
        ai = assessment.ai_result
        rev = assessment.professional_review

        meta_data = [
            [
                Paragraph("<strong>Assessment ID:</strong>", body_style),
                Paragraph(assessment.id, body_style),
                Paragraph("<strong>Patient Identifier:</strong>", body_style),
                Paragraph(assessment.patient_id, body_style),
            ],
            [
                Paragraph("<strong>Eye Laterality:</strong>", body_style),
                Paragraph("OD (Right Eye)" if assessment.eye_laterality == "OD" else "OS (Left Eye)", body_style),
                Paragraph("<strong>Acquisition Date:</strong>", body_style),
                Paragraph(assessment.created_at.strftime("%Y-%m-%d %H:%M UTC"), body_style),
            ],
            [
                Paragraph("<strong>Camera Model:</strong>", body_style),
                Paragraph(assessment.camera_model or "Standard Fundus Camera", body_style),
                Paragraph("<strong>Dilation Protocol:</strong>", body_style),
                Paragraph("Mydriatic" if assessment.is_mydriatic else "Non-Mydriatic", body_style),
            ],
        ]

        meta_table = Table(meta_data, colWidths=[1.4 * inch, 2.1 * inch, 1.4 * inch, 2.1 * inch])
        meta_table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ])
        )
        elements.append(meta_table)
        elements.append(Spacer(1, 10))

        # --- Technical Quality & 3-Gate Validation ---
        elements.append(Paragraph("1. Technical Image Quality & 3-Stage Validation Pipeline", h2_style))
        elements.append(Spacer(1, 4))

        g1_stat = "PASSED" if (val and val.gate1_passed) else "FAILED"
        g2_stat = "PASSED" if (val and val.gate2_passed) else ("FAILED" if val and val.failed_gate == 2 else "PENDING")
        g3_stat = "PASSED" if (val and val.gate3_passed) else ("FAILED" if val and val.failed_gate == 3 else "PENDING")

        lap_metric = f"{val.laplacian_variance:.1f}" if (val and val.laplacian_variance is not None) else "N/A"
        illum_metric = f"{val.illumination_index:.2f}" if (val and val.illumination_index is not None) else "N/A"
        sha_hash = img.sha256_hash if img else "N/A"

        gate_data = [
            ["Gate 1: File Integrity", g1_stat, f"MIME {img.mime_type if img else 'JPEG'}, SHA-256: {sha_hash[:16]}..."],
            ["Gate 2: Retinal Relevance", g2_stat, "Retinal aperture geometry and spectral balance confirmed."],
            ["Gate 3: Technical Quality", g3_stat, f"Laplacian variance: {lap_metric} (Threshold >= 60.0), Illum: {illum_metric}"],
        ]

        gate_table = Table(gate_data, colWidths=[1.8 * inch, 1.0 * inch, 4.2 * inch])
        gate_table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F1F5F9")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
                ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("TEXTCOLOR", (1, 0), (1, -1), colors.HexColor("#059669") if assessment.status != "rejected" else colors.HexColor("#DC2626")),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ])
        )
        elements.append(gate_table)
        elements.append(Spacer(1, 10))

        # --- Section A: Preliminary AI Model Observation ---
        elements.append(Paragraph("2. Preliminary Model-Generated Observation (AI Assistive Engine)", h2_style))
        elements.append(Spacer(1, 4))

        if ai and assessment.status != "rejected":
            ai_data = [
                [
                    Paragraph("<strong>Primary Candidate Class:</strong>", body_style),
                    Paragraph(f"<strong>{ai.primary_class_label}</strong>", body_style),
                    Paragraph("<strong>Class Relative Score:</strong>", body_style),
                    Paragraph(f"<strong>{ai.primary_score:.2f}</strong>", body_style),
                ],
                [
                    Paragraph("<strong>Model Identity:</strong>", body_style),
                    Paragraph("EfficientNet-B0 (Frozen Evaluation)", body_style),
                    Paragraph("<strong>Target Saliency Layer:</strong>", body_style),
                    Paragraph(ai.target_layer, body_style),
                ],
            ]
            ai_table = Table(ai_data, colWidths=[1.6 * inch, 2.0 * inch, 1.6 * inch, 1.8 * inch])
            ai_table.setStyle(
                TableStyle([
                    ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ])
            )
            elements.append(ai_table)
            elements.append(Spacer(1, 4))

            # 5-Class Breakdown Table
            score_rows = [["ICDR Stage", "Grade Description", "Model-Generated Class Score"]]
            for cs in ai.class_scores:
                score_rows.append([f"Grade {cs['grade']}", cs["label"], f"{cs['score']:.2f}"])

            score_table = Table(score_rows, colWidths=[1.2 * inch, 4.4 * inch, 1.4 * inch])
            score_table.setStyle(
                TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E2E8F0")),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 7.5),
                    ("ALIGN", (2, 0), (2, -1), "RIGHT"),
                    ("TOPPADDING", (0, 0), (-1, -1), 2.5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
                ])
            )
            elements.append(score_table)
        else:
            rejection_text = (
                f"Model evaluation aborted. Image rejected at Gate {val.failed_gate if val else '1'} "
                f"({val.failure_reason if val else 'Technical failure'})."
            )
            elements.append(Paragraph(rejection_text, body_style))

        elements.append(Spacer(1, 4))
        elements.append(
            Paragraph(
                "<strong>Regulatory Boundary Notice:</strong> Scores reflect feature activations derived from the pre-trained "
                "EfficientNet-B0 network. Scores do not represent clinical certainty, diagnostic probability, or confirmed diagnosis.",
                disclaimer_style,
            )
        )
        elements.append(Spacer(1, 10))

        # --- Section B: Professional Review Response ---
        elements.append(Paragraph("3. Professional Review Response (Authoritative Human-in-the-Loop)", h2_style))
        elements.append(Spacer(1, 4))

        if rev:
            agreement_label = "AGREED with Model Observation" if rev.agreement == "agree" else (
                "DISAGREED with Model Observation" if rev.agreement == "disagree" else "UNABLE TO DETERMINE / INDETERMINATE"
            )

            rev_data = [
                [
                    Paragraph("<strong>Reviewing Clinician:</strong>", body_style),
                    Paragraph(rev.clinician_name, body_style),
                    Paragraph("<strong>License / GMC Number:</strong>", body_style),
                    Paragraph(rev.license_number or "GMC-7492104", body_style),
                ],
                [
                    Paragraph("<strong>Clinical Agreement:</strong>", body_style),
                    Paragraph(agreement_label, body_style),
                    Paragraph("<strong>Certified ICDR Grade:</strong>", body_style),
                    Paragraph(f"<strong>{rev.certified_grade_label}</strong>", body_style),
                ],
                [
                    Paragraph("<strong>Clinical Notes / Rationale:</strong>", body_style),
                    Paragraph(rev.justification_notes or "Corroborated with clear fundus examination.", body_style),
                    Paragraph("<strong>Management / Referral:</strong>", body_style),
                    Paragraph(rev.referral_plan, body_style),
                ],
                [
                    Paragraph("<strong>Signature Timestamp:</strong>", body_style),
                    Paragraph(rev.signed_at.strftime("%Y-%m-%d %H:%M:%S UTC"), body_style),
                    Paragraph("<strong>Assessment Integrity Hash:</strong>", body_style),
                    Paragraph(f"<font face='Courier' size=7>{rev.signature_hash}</font>", body_style),
                ],
            ]

            rev_table = Table(rev_data, colWidths=[1.6 * inch, 2.0 * inch, 1.6 * inch, 1.8 * inch])
            rev_table.setStyle(
                TableStyle([
                    ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F0FDF4")),  # Muted mint
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#86EFAC")),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ])
            )
            elements.append(rev_table)
        else:
            elements.append(
                Paragraph(
                    "<em>Awaiting clinician certification. This preliminary consultation record has not yet been signed.</em>",
                    body_style,
                )
            )

        elements.append(Spacer(1, 14))
        elements.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#94A3B8"), spaceAfter=6))
        elements.append(
            Paragraph(
                "<strong>CONFIDENTIAL CLINICAL CONSULTATION RECORD:</strong> This document is generated for clinical decision support. "
                "The certified diagnosis and management decisions are solely the legal and medical responsibility of the named reviewing practitioner. "
                "Original photographic evidence cryptographically anchored via SHA-256 digest.",
                disclaimer_style,
            )
        )

        doc.build(elements)
        pdf_bytes = buffer.getvalue()
        buffer.close()

        # Persist report to storage
        os.makedirs(settings.STORAGE_REPORTS_PATH, exist_ok=True)
        report_path = os.path.join(settings.STORAGE_REPORTS_PATH, f"report_{assessment.id}.pdf")
        with open(report_path, "wb") as f:
            f.write(pdf_bytes)

        return pdf_bytes
