"""
Report Service
==============
Generates PDF experiment reports using ReportLab.
TXT, CSV, JSON, and PDF export formats are supported.

Reports contain only real saved experiment data.
If an experiment has not been run, the report is labeled as a template.
API tokens and secrets are NEVER included in reports.
"""

import csv
import io
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        HRFlowable, PageBreak
    )
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    logger.warning("ReportLab not installed. PDF export disabled.")


def export_txt(bitstream: str) -> bytes:
    """Export raw bitstream as UTF-8 text."""
    return bitstream.encode("utf-8")


def export_json(experiment: Any, debias: Any = None, test_runs: Any = None) -> bytes:
    """
    Export experiment data as structured JSON.
    API tokens and secrets are excluded.
    """
    data: Dict[str, Any] = {
        "export_format": "QRNG Experiment JSON v1.0",
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "experiment": {
            "experiment_id": experiment.experiment_id,
            "created_at": experiment.created_at.isoformat() if experiment.created_at else None,
            "source_type": experiment.source_type,
            "backend_name": experiment.backend_name,
            "requested_bits": experiment.requested_bits,
            "actual_bits": experiment.actual_bits,
            "seed_used": experiment.seed_used,
            "is_seeded": experiment.is_seeded,
            "circuit_description": experiment.circuit_description,
            "noise_params": experiment.noise_params,
            "execution_time_seconds": experiment.execution_time_seconds,
            "status": experiment.status,
            "measurement_counts": experiment.measurement_counts,
            "zeros": experiment.zeros,
            "ones": experiment.ones,
            "proportion_ones": experiment.proportion_ones,
            "shannon_entropy": experiment.shannon_entropy,
            "min_entropy": getattr(experiment, "min_entropy", None),
            "bias_deviation": experiment.bias_deviation,
            "num_qubits": getattr(experiment, "num_qubits", 1),
            "source_classification_note": experiment.source_classification_note,
            "warnings": experiment.warnings,
            # bitstream is large; stored separately to keep JSON readable
            "bitstream_length": len(experiment.bitstream) if experiment.bitstream else 0,
        },
        "debiasing": None,
        "statistical_tests": [],
        "note": (
            "API tokens and secrets are never included in exports. "
            "Bitstream is omitted from JSON metadata; use TXT export for raw bits."
        ),
    }

    if debias:
        data["debiasing"] = {
            "debias_id": debias.debias_id,
            "input_bits": debias.input_bits,
            "output_bits": debias.output_bits,
            "count_00": debias.count_00,
            "count_01": debias.count_01,
            "count_10": debias.count_10,
            "count_11": debias.count_11,
            "retained_pairs": debias.retained_pairs,
            "discarded_pairs": debias.discarded_pairs,
            "discarded_bits": getattr(debias, "discarded_bits", None),
            "output_to_input_ratio": getattr(debias, "output_to_input_ratio", debias.extraction_efficiency),
            "pair_retention_ratio": getattr(debias, "pair_retention_ratio", None),
            "extraction_efficiency": debias.extraction_efficiency,
            "output_proportion_ones": debias.output_proportion_ones,
            "output_min_entropy": getattr(debias, "output_min_entropy", None),
        }

    if test_runs:
        for run in test_runs:
            data["statistical_tests"].append({
                "test_run_id": run.test_run_id,
                "tested_on": run.tested_on,
                "bits_analyzed": run.bits_analyzed,
                "significance_level": run.significance_level,
                "passed": run.passed,
                "failed": run.failed,
                "not_run": run.not_run,
                "results": run.results_json,
            })

    return json.dumps(data, indent=2, default=str).encode("utf-8")


def export_csv(experiment: Any, debias: Any = None, test_runs: Any = None) -> bytes:
    """Export summary metrics as CSV."""
    buf = io.StringIO()
    writer = csv.writer(buf)

    writer.writerow(["# QRNG Experiment CSV Export"])
    writer.writerow(["# Generated:", datetime.utcnow().isoformat() + "Z"])
    writer.writerow([])

    # Experiment summary
    writer.writerow(["Section", "Field", "Value"])
    writer.writerow(["Experiment", "experiment_id", experiment.experiment_id])
    writer.writerow(["Experiment", "source_type", experiment.source_type])
    writer.writerow(["Experiment", "backend_name", experiment.backend_name])
    writer.writerow(["Experiment", "requested_bits", experiment.requested_bits])
    writer.writerow(["Experiment", "actual_bits", experiment.actual_bits])
    writer.writerow(["Experiment", "zeros", experiment.zeros])
    writer.writerow(["Experiment", "ones", experiment.ones])
    writer.writerow(["Experiment", "proportion_ones", experiment.proportion_ones])
    writer.writerow(["Experiment", "shannon_entropy_per_bit", experiment.shannon_entropy])
    writer.writerow(["Experiment", "min_entropy_per_bit", getattr(experiment, "min_entropy", None)])
    writer.writerow(["Experiment", "bias_deviation", experiment.bias_deviation])
    writer.writerow(["Experiment", "num_qubits", getattr(experiment, "num_qubits", 1)])
    writer.writerow(["Experiment", "execution_time_s", experiment.execution_time_seconds])
    writer.writerow(["Experiment", "is_seeded", experiment.is_seeded])
    writer.writerow(["Experiment", "seed_used", experiment.seed_used])
    writer.writerow([])

    if debias:
        writer.writerow(["Debiasing", "input_bits", debias.input_bits])
        writer.writerow(["Debiasing", "output_bits", debias.output_bits])
        writer.writerow(["Debiasing", "discarded_bits", getattr(debias, "discarded_bits", None)])
        writer.writerow(["Debiasing", "count_00", debias.count_00])
        writer.writerow(["Debiasing", "count_01", debias.count_01])
        writer.writerow(["Debiasing", "count_10", debias.count_10])
        writer.writerow(["Debiasing", "count_11", debias.count_11])
        writer.writerow(["Debiasing", "retained_pairs", debias.retained_pairs])
        writer.writerow(["Debiasing", "discarded_pairs", debias.discarded_pairs])
        writer.writerow(["Debiasing", "output_to_input_ratio", getattr(debias, "output_to_input_ratio", debias.extraction_efficiency)])
        writer.writerow(["Debiasing", "pair_retention_ratio", getattr(debias, "pair_retention_ratio", None)])
        writer.writerow(["Debiasing", "extraction_efficiency", debias.extraction_efficiency])
        writer.writerow(["Debiasing", "output_min_entropy", getattr(debias, "output_min_entropy", None)])
        writer.writerow([])

    if test_runs:
        for run in test_runs:
            writer.writerow([
                "Test Run", "test_run_id", run.test_run_id,
                "tested_on", run.tested_on, "bits_analyzed", run.bits_analyzed,
            ])
            if run.results_json:
                writer.writerow([
                    "Test", "name", "outcome", "statistic", "p_value", "threshold"
                ])
                for r in run.results_json:
                    writer.writerow([
                        "Test",
                        r.get("test_name", ""),
                        r.get("outcome", ""),
                        r.get("statistic", ""),
                        r.get("p_value", ""),
                        r.get("significance_level", ""),
                    ])
            writer.writerow([])

    return buf.getvalue().encode("utf-8")


def export_pdf(
    experiment: Any,
    debias: Any = None,
    test_runs: Any = None,
    reports_dir: str = "reports",
) -> bytes:
    """
    Generate a PDF experiment report using ReportLab.
    Contains only real saved experiment data.
    API tokens and secrets are NEVER included.
    """
    if not REPORTLAB_AVAILABLE:
        raise RuntimeError(
            "ReportLab is not installed. Install it with: pip install reportlab"
        )

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        rightMargin=2 * cm,
        leftMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
    )

    styles = getSampleStyleSheet()
    style_title = ParagraphStyle(
        "Title2", parent=styles["Title"], fontSize=18, spaceAfter=6,
        textColor=colors.HexColor("#1a1a2e")
    )
    style_h2 = ParagraphStyle(
        "Heading2a", parent=styles["Heading2"], fontSize=13, spaceAfter=4,
        textColor=colors.HexColor("#16213e")
    )
    style_h3 = ParagraphStyle(
        "Heading3a", parent=styles["Heading3"], fontSize=11, spaceAfter=3,
        textColor=colors.HexColor("#0f3460")
    )
    style_body = ParagraphStyle(
        "Body", parent=styles["Normal"], fontSize=9, leading=13,
        spaceAfter=4, alignment=TA_JUSTIFY
    )
    style_note = ParagraphStyle(
        "Note", parent=styles["Normal"], fontSize=8, leading=11,
        textColor=colors.HexColor("#555555"), spaceAfter=4
    )
    style_warning = ParagraphStyle(
        "Warning", parent=styles["Normal"], fontSize=8, leading=11,
        textColor=colors.HexColor("#c0392b"), spaceAfter=4
    )
    style_center = ParagraphStyle(
        "Center", parent=styles["Normal"], fontSize=9, alignment=TA_CENTER
    )

    story = []
    hr = HRFlowable(width="100%", thickness=1, color=colors.HexColor("#0f3460"))

    # ── Cover ─────────────────────────────────────────────────────────────────
    story.append(Spacer(1, 1 * cm))
    story.append(Paragraph(
        "Quantum Random Number Generator (QRNG)", style_title
    ))
    story.append(Paragraph(
        "with Debiasing and Statistical Randomness Analysis", style_h2
    ))
    story.append(Spacer(1, 0.3 * cm))
    story.append(hr)
    story.append(Spacer(1, 0.3 * cm))
    story.append(Paragraph(
        f"<b>Course:</b> Quantum Computing (25AM3504) &nbsp;|&nbsp; <b>Project #15</b>",
        style_body
    ))
    story.append(Paragraph(
        f"<b>Experiment ID:</b> {experiment.experiment_id}", style_body
    ))
    story.append(Paragraph(
        f"<b>Report Generated:</b> {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC",
        style_body
    ))
    story.append(Paragraph(
        f"<b>Source Classification:</b> {experiment.source_type.upper()}", style_body
    ))
    story.append(Spacer(1, 0.5 * cm))
    story.append(Paragraph(
        "<b>⚠ Source Disclaimer:</b> " + (experiment.source_classification_note or ""),
        style_warning
    ))
    story.append(Spacer(1, 0.5 * cm))

    # ── Experiment Details ─────────────────────────────────────────────────────
    story.append(Paragraph("1. Experiment Configuration", style_h2))
    story.append(hr)
    story.append(Spacer(1, 0.2 * cm))

    exp_data = [
        ["Field", "Value"],
        ["Backend", experiment.backend_name],
        ["Qubit Count", str(getattr(experiment, "num_qubits", 1) or 1)],
        ["Requested Bits", str(experiment.requested_bits)],
        ["Actual Bits", str(experiment.actual_bits)],
        ["Seeded Run", "Yes" if experiment.is_seeded else "No"],
        ["Seed Value", str(experiment.seed_used) if experiment.seed_used else "N/A"],
        ["Execution Time", f"{experiment.execution_time_seconds:.4f} s"],
        ["Circuit", experiment.circuit_description or "N/A"],
    ]
    if experiment.noise_params:
        for k, v in experiment.noise_params.items():
            if k != "note":
                exp_data.append([f"  {k}", str(v)])

    t = Table(exp_data, colWidths=[5 * cm, 11.5 * cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f3460")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f0f4ff")]),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#cccccc")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(t)
    story.append(Spacer(1, 0.5 * cm))

    # ── Bit Distribution ───────────────────────────────────────────────────────
    story.append(Paragraph("2. Bit Distribution Statistics", style_h2))
    story.append(hr)
    story.append(Spacer(1, 0.2 * cm))

    min_ent_str = f"{experiment.min_entropy:.6f} bits/symbol" if getattr(experiment, "min_entropy", None) is not None else "N/A"
    bit_data = [
        ["Metric", "Value"],
        ["Total Bits", str(experiment.actual_bits)],
        ["Count of 0s", str(experiment.zeros)],
        ["Count of 1s", str(experiment.ones)],
        ["Proportion of 1s", f"{experiment.proportion_ones:.6f}"],
        ["Absolute Deviation from 0.5", f"{experiment.bias_deviation:.6f}"],
        ["Shannon Entropy (per bit)", f"{experiment.shannon_entropy:.6f} bits/symbol"],
        ["Empirical Min-Entropy (H_min)", min_ent_str],
    ]
    t2 = Table(bit_data, colWidths=[8 * cm, 8.5 * cm])
    t2.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#16213e")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f0f4ff")]),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#cccccc")),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(t2)
    story.append(Spacer(1, 0.3 * cm))
    story.append(Paragraph(
        "Note: Shannon entropy and empirical min-entropy are sample descriptive statistics. "
        "They do NOT certify cryptographic entropy or prove true randomness.",
        style_note
    ))
    story.append(Spacer(1, 0.5 * cm))

    # ── Debiasing ─────────────────────────────────────────────────────────────
    if debias:
        story.append(Paragraph("3. Von Neumann Debiasing Results", style_h2))
        story.append(hr)
        story.append(Spacer(1, 0.2 * cm))

        out_to_in = getattr(debias, "output_to_input_ratio", debias.extraction_efficiency)
        pair_ret = getattr(debias, "pair_retention_ratio", None)
        pair_ret_str = f"{pair_ret:.4f} ({pair_ret*100:.2f}%)" if pair_ret is not None else "N/A"
        disc_bits_str = str(getattr(debias, "discarded_bits", debias.input_bits - debias.output_bits))

        db_data = [
            ["Metric", "Value"],
            ["Input Bits", str(debias.input_bits)],
            ["Complete Pairs Processed", str(debias.total_complete_pairs)],
            ["Pairs 00 (discarded)", str(debias.count_00)],
            ["Pairs 01 → output 0", str(debias.count_01)],
            ["Pairs 10 → output 1", str(debias.count_10)],
            ["Pairs 11 (discarded)", str(debias.count_11)],
            ["Retained Pairs", str(debias.retained_pairs)],
            ["Discarded Pairs", str(debias.discarded_pairs)],
            ["Total Discarded Bits", disc_bits_str],
            ["Output Bits", str(debias.output_bits)],
            ["Output Proportion of 1s", f"{debias.output_proportion_ones:.6f}" if debias.output_proportion_ones else "N/A"],
            ["Output-to-Input Ratio", f"{out_to_in:.4f} ({out_to_in*100:.2f}%) [Expected: ~25%]"],
            ["Pair-Retention Ratio", f"{pair_ret_str} [Expected: ~50%]"],
        ]
        t3 = Table(db_data, colWidths=[8 * cm, 8.5 * cm])
        t3.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f3460")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f0f4ff")]),
            ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#cccccc")),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]))
        story.append(t3)
        story.append(Spacer(1, 0.3 * cm))
        story.append(Paragraph(
            "Algorithm: Von Neumann Extractor. Pairs: 01→0, 10→1, 00 and 11 discarded. "
            "Assumes i.i.d. input bits. Does NOT remove arbitrary correlations or prove randomness.",
            style_note
        ))
        story.append(Spacer(1, 0.5 * cm))

    # ── Statistical Tests ─────────────────────────────────────────────────────
    if test_runs:
        story.append(Paragraph("4. Selected NIST SP 800-22 Statistical Test Results", style_h2))
        story.append(hr)
        story.append(Spacer(1, 0.2 * cm))
        story.append(Paragraph(
            "SELECTED SUBSET — NOT the complete NIST SP 800-22 suite (15 tests). "
            "Tests implemented: Frequency, Block Frequency, Runs, Longest Run, Spectral (5 of 15).",
            style_warning
        ))
        story.append(Spacer(1, 0.2 * cm))

        for run in test_runs:
            story.append(Paragraph(
                f"Test Run on: {run.tested_on.upper()} stream | "
                f"Bits analyzed: {run.bits_analyzed} | α = {run.significance_level}",
                style_h3
            ))
            if run.results_json:
                hdr = ["Test", "Outcome", "Statistic", "p-value"]
                rows = [hdr]
                for r in run.results_json:
                    rows.append([
                        r.get("test_name", "")[:35],
                        r.get("outcome", ""),
                        f"{r.get('statistic', ''):.4f}" if r.get('statistic') is not None else "N/A",
                        f"{r.get('p_value', ''):.4f}" if r.get('p_value') is not None else "N/A",
                    ])
                t4 = Table(rows, colWidths=[7 * cm, 2.5 * cm, 2.5 * cm, 2.5 * cm])
                outcome_colors = {"PASS": colors.HexColor("#27ae60"),
                                  "FAIL": colors.HexColor("#e74c3c"),
                                  "NOT_RUN": colors.HexColor("#7f8c8d"),
                                  "INSUFFICIENT_DATA": colors.HexColor("#e67e22"),
                                  "ERROR": colors.HexColor("#8e44ad")}
                ts = TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#16213e")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 7),
                    ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#cccccc")),
                    ("TOPPADDING", (0, 0), (-1, -1), 2),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
                ])
                for i, r in enumerate(run.results_json, start=1):
                    oc = r.get("outcome", "")
                    col = outcome_colors.get(oc, colors.white)
                    ts.add("TEXTCOLOR", (1, i), (1, i), col)
                    ts.add("FONTNAME", (1, i), (1, i), "Helvetica-Bold")
                t4.setStyle(ts)
                story.append(t4)
            story.append(Paragraph(
                f"Passed: {run.passed} | Failed: {run.failed} | Not run: {run.not_run}",
                style_body
            ))
            story.append(Spacer(1, 0.3 * cm))

        story.append(Paragraph(
            "IMPORTANT: Passing statistical tests does NOT prove true randomness "
            "or cryptographic security. A p-value is NOT the probability the data is random.",
            style_warning
        ))
        story.append(Spacer(1, 0.5 * cm))

    # ── Limitations ───────────────────────────────────────────────────────────
    story.append(Paragraph("5. Limitations and Scientific Interpretation", style_h2))
    story.append(hr)
    story.append(Spacer(1, 0.2 * cm))
    limitations = [
        "• Simulator modes use classical computation and pseudo-random sampling. "
          "They are NOT sources of physical quantum randomness.",
        "• The simplified readout-error noise model does NOT reproduce any specific "
          "real quantum device perfectly.",
        "• Aggregated measurement counts do NOT preserve the chronological order of shots.",
        "• Shannon entropy is a descriptive statistic; it does NOT certify cryptographic entropy.",
        "• The von Neumann extractor assumes i.i.d. input bits. It does NOT remove "
          "arbitrary correlations.",
        "• Statistical tests assess selected statistical properties only. "
          "Passing them does NOT prove true randomness.",
        "• This application is NOT certified as a cryptographically secure random number generator.",
        "• Hardware results (when available) reflect real qubit measurements but are "
          "affected by device errors, decoherence, and calibration imperfections.",
    ]
    for lim in limitations:
        story.append(Paragraph(lim, style_note))
    story.append(Spacer(1, 0.5 * cm))

    # ── References ────────────────────────────────────────────────────────────
    story.append(Paragraph("6. References", style_h2))
    story.append(hr)
    story.append(Spacer(1, 0.2 * cm))
    refs = [
        "[1] Bassham, L. et al. (2010). NIST SP 800-22 Rev. 1a: A Statistical Test Suite "
            "for Random and Pseudorandom Number Generators for Cryptographic Applications. "
            "NIST. https://nvlpubs.nist.gov/nistpubs/Legacy/SP/nistspecialpublication800-22r1a.pdf",
        "[2] Qiskit contributors (2023). Qiskit: An Open-source Framework for Quantum Computing. "
            "https://qiskit.org/documentation/",
        "[3] Qiskit Aer contributors (2023). Qiskit Aer Simulator. "
            "https://qiskit.github.io/qiskit-aer/",
        "[4] Von Neumann, J. (1951). Various techniques used in connection with random digits. "
            "Applied Mathematics Series, 12, 36–38. National Bureau of Standards.",
        "[5] Nielsen, M. A., & Chuang, I. L. (2010). Quantum Computation and Quantum Information. "
            "Cambridge University Press.",
    ]
    for ref in refs:
        story.append(Paragraph(ref, style_note))
        story.append(Spacer(1, 0.1 * cm))

    story.append(Spacer(1, 0.5 * cm))
    story.append(Paragraph(
        "This report was generated automatically by the QRNG application. "
        "No API tokens or secrets are included in this document.",
        style_note
    ))

    doc.build(story)
    return buf.getvalue()
