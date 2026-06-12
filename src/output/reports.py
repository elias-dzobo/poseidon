"""
Flood report output writer.

Produces:
  - JSON report (machine-readable, matches FloodReport schema)
  - Plain-text situation report (human-readable, for alerts/email)
  - PDF report (formatted, for stakeholders) [future]
"""

from __future__ import annotations

import logging
from pathlib import Path

from src.models.vlm.model import FloodReport
from src.models.vlm.generate import report_to_json, report_to_text

logger = logging.getLogger(__name__)


class ReportWriter:
    """
    Serialises FloodReport objects to various output formats.

    Parameters
    ----------
    output_dir : str | Path
        Directory for report outputs.
    """

    def __init__(self, output_dir: str | Path = "outputs/reports") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write_json(self, report: FloodReport) -> Path:
        """
        Write the flood report as a JSON file.

        Returns
        -------
        Path
            Path to the written JSON file.
        """
        output_path = self.output_dir / f"{report.event_id}.json"
        output_path.write_text(report_to_json(report), encoding="utf-8")
        logger.info(f"JSON report written: {output_path}")
        return output_path

    def write_text(self, report: FloodReport) -> Path:
        """
        Write the flood report as a plain-text situation report.

        Returns
        -------
        Path
            Path to the written .txt file.
        """
        output_path = self.output_dir / f"{report.event_id}.txt"
        output_path.write_text(report_to_text(report), encoding="utf-8")
        logger.info(f"Text report written: {output_path}")
        return output_path

    def write_pdf(self, report: FloodReport) -> Path:
        """
        Render the flood report as a formatted PDF.

        Includes:
          - Report header with event metadata
          - Severity indicator
          - Flood extent map thumbnail
          - Affected areas table
          - Recommendations list

        Returns
        -------
        Path
            Path to the written PDF file.
        """
        output_path = self.output_dir / f"{report.event_id}.pdf"
        # TODO: use reportlab or weasyprint to render HTML template → PDF
        raise NotImplementedError
