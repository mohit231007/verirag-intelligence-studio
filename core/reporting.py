"""Portable PDF export for an answer and its evidence trail."""

from __future__ import annotations

from io import BytesIO
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from .models import QueryTrace


def _safe_paragraph(text: str) -> str:
    return escape(text).replace("\n", "<br/>")


def build_answer_pdf(trace: QueryTrace) -> bytes:
    """Create a compact answer report containing provenance and query diagnostics."""

    output = BytesIO()
    document = SimpleDocTemplate(
        output,
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=17 * mm,
        bottomMargin=17 * mm,
        title="VeriRAG Answer Report",
        author="VeriRAG Studio",
    )
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            "Brand",
            parent=styles["Title"],
            textColor=colors.HexColor("#0B6357"),
            alignment=TA_CENTER,
            spaceAfter=4 * mm,
        )
    )
    styles.add(
        ParagraphStyle(
            "Evidence",
            parent=styles["BodyText"],
            backColor=colors.HexColor("#EEF8F6"),
            borderColor=colors.HexColor("#A4D9D1"),
            borderWidth=0.5,
            borderPadding=7,
            leading=14,
            spaceAfter=4 * mm,
        )
    )
    styles.add(
        ParagraphStyle(
            "ChunkCaption",
            parent=styles["BodyText"],
            textColor=colors.HexColor("#5B6B73"),
            fontSize=8,
            leading=10,
        )
    )

    story = [
        Paragraph("VeriRAG Studio", styles["Brand"]),
        Paragraph("Verified Answer Report", styles["Heading2"]),
        Spacer(1, 2 * mm),
        Paragraph("Question", styles["Heading3"]),
        Paragraph(_safe_paragraph(trace.query), styles["BodyText"]),
        Spacer(1, 3 * mm),
        Paragraph("Answer", styles["Heading3"]),
        Paragraph(_safe_paragraph(trace.answer), styles["BodyText"]),
        Spacer(1, 5 * mm),
    ]

    summary = [
        ["Outcome", "Safe refusal" if trace.is_refusal else "Citation-validated"],
        ["Confidence", trace.confidence],
        ["Provider / model", f"{trace.provider} / {trace.model}"],
        ["Total latency", f"{trace.total_ms / 1_000:.2f} seconds"],
        ["Standalone retrieval query", trace.standalone_query],
    ]
    table = Table(summary, colWidths=[45 * mm, 112 * mm])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#123047")),
                ("TEXTCOLOR", (0, 0), (0, -1), colors.white),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#CAD8DF")),
                ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    story.extend([table, PageBreak(), Paragraph("Evidence trail", styles["Heading2"])])

    if not trace.retrieved:
        story.append(Paragraph("No evidence passages crossed the configured gate.", styles["BodyText"]))
    for source_number, item in enumerate(trace.retrieved, start=1):
        heading = (
            f"S{source_number} · {escape(item.chunk.source_doc)} · page {item.chunk.page_number} · "
            f"similarity {item.similarity:.1%}"
        )
        story.append(Paragraph(heading, styles["Heading3"]))
        story.append(Paragraph(_safe_paragraph(item.chunk.text), styles["Evidence"]))
        story.append(Paragraph(f"Chunk ID: {escape(item.chunk.chunk_id)}", styles["ChunkCaption"]))
        story.append(Spacer(1, 3 * mm))

    document.build(story)
    return output.getvalue()
