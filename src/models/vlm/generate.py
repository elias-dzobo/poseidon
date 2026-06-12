"""
Flood report generation utilities.

Handles:
  - Formatting FloodReport dataclasses into JSON / human-readable text
  - Building the metadata dict passed to LFMFloodReporter.generate()
  - Determining flood season from acquisition date (Ghana-specific)
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from src.models.vlm.model import FloodReport


# Ghana's two rainy seasons (approximate)
GHANA_RAINY_SEASONS = [
    (4, 6),   # April–June (major)
    (9, 11),  # September–November (minor)
]


def get_ghana_season(date: datetime) -> str:
    """
    Classify acquisition date into Ghana's seasonal calendar.

    Returns
    -------
    str
        One of: 'major_rainy', 'minor_rainy', 'dry'.
    """
    month = date.month
    for start, end in GHANA_RAINY_SEASONS:
        if start <= month <= end:
            label = "major_rainy" if start == 4 else "minor_rainy"
            return label
    return "dry"


def build_metadata(
    acquisition_date: str | datetime,
    region: dict,
    data_sources: list[str],
) -> dict:
    """
    Assemble the metadata dict passed to the VLM prompt.

    Parameters
    ----------
    acquisition_date : str | datetime
        Scene acquisition datetime.
    region : dict
        Region dict: {country, city, aoi_bbox}.
    data_sources : list[str]
        E.g. ["Sentinel-1 IW GRD", "Sentinel-2 L2A"].

    Returns
    -------
    dict
        Metadata dict ready for LFMFloodReporter.generate().
    """
    if isinstance(acquisition_date, str):
        acquisition_date = datetime.fromisoformat(acquisition_date)

    return {
        "acquisition_date": acquisition_date.isoformat(),
        "region": region,
        "season": get_ghana_season(acquisition_date),
        "data_sources": data_sources,
    }


def report_to_json(report: FloodReport, indent: int = 2) -> str:
    """Serialise a FloodReport to a JSON string."""
    return json.dumps(
        {
            "event_id": report.event_id,
            "acquisition_date": report.acquisition_date,
            "region": report.region,
            "severity": report.severity,
            "confidence_score": report.confidence_score,
            "flood_extent": {
                "total_area_km2": report.flood_extent.total_area_km2,
                "pct_of_aoi_flooded": report.flood_extent.pct_of_aoi_flooded,
                "change_from_baseline_km2": report.flood_extent.change_from_baseline_km2,
            },
            "affected_areas": [
                {
                    "name": a.name,
                    "severity": a.severity,
                    "area_km2": a.area_km2,
                }
                for a in report.affected_areas
            ],
            "hydrological_indicators": {
                "odaw_river_status": report.hydrological_indicators.odaw_river_status,
                "korle_lagoon_status": report.hydrological_indicators.korle_lagoon_status,
                "permanent_water_expansion_pct": (
                    report.hydrological_indicators.permanent_water_expansion_pct
                ),
            },
            "recommendations": report.recommendations,
            "data_sources": report.data_sources,
            "model_versions": report.model_versions,
        },
        indent=indent,
    )


def report_to_text(report: FloodReport) -> str:
    """
    Format a FloodReport as a human-readable situation report.

    Suitable for email alerts and PDF export.
    """
    severity_emoji = {
        "low": "🟡",
        "medium": "🟠",
        "high": "🔴",
        "critical": "🚨",
    }

    lines = [
        f"FLOOD SITUATION REPORT — {report.region.get('city', 'Unknown')}, "
        f"{report.region.get('country', 'Unknown')}",
        f"Event ID   : {report.event_id}",
        f"Date       : {report.acquisition_date}",
        f"Severity   : {severity_emoji.get(report.severity, '')} {report.severity.upper()}",
        f"Confidence : {report.confidence_score:.0%}",
        "",
        "FLOOD EXTENT",
        f"  Total inundated area : {report.flood_extent.total_area_km2:.1f} km²",
        f"  AOI coverage         : {report.flood_extent.pct_of_aoi_flooded:.1f}%",
        f"  Change from baseline : +{report.flood_extent.change_from_baseline_km2:.1f} km²",
        "",
        "AFFECTED AREAS",
    ]

    for area in report.affected_areas:
        lines.append(
            f"  {area.name:<20} {area.severity.upper():<10} {area.area_km2:.2f} km²"
        )

    lines += [
        "",
        "HYDROLOGICAL STATUS",
        f"  Odaw River     : {report.hydrological_indicators.odaw_river_status}",
        f"  Korle Lagoon   : {report.hydrological_indicators.korle_lagoon_status}",
        f"  Water expansion: +{report.hydrological_indicators.permanent_water_expansion_pct:.0f}%",
        "",
        "RECOMMENDATIONS",
    ]

    for i, rec in enumerate(report.recommendations, 1):
        lines.append(f"  {i}. {rec}")

    lines += [
        "",
        f"Data sources : {', '.join(report.data_sources)}",
        f"Models       : {report.model_versions}",
    ]

    return "\n".join(lines)
