"""
Flood map output writer.

Produces:
  - GeoTIFF flood mask annotated with event metadata
  - GeoJSON flood polygons with per-feature severity attributes
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from src.models.vlm.model import FloodReport

logger = logging.getLogger(__name__)


class FloodMapWriter:
    """
    Writes flood extent outputs (GeoTIFF + GeoJSON) from mask + report.

    Parameters
    ----------
    output_dir : str | Path
        Directory for map outputs.
    """

    def __init__(self, output_dir: str | Path = "outputs/maps") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write_geotiff(self, mask_path: Path, report: FloodReport) -> Path:
        """
        Copy the binary flood mask to the output directory, annotating
        the GeoTIFF metadata tags with event_id, severity, and date.

        Parameters
        ----------
        mask_path : Path
            Binary flood mask GeoTIFF from U-Net.
        report : FloodReport
            Parsed flood report (for metadata tags).

        Returns
        -------
        Path
            Path to the annotated output GeoTIFF.
        """
        output_path = self.output_dir / f"{report.event_id}_flood_mask.tif"
        # TODO: open mask_path with rasterio, update profile tags with
        #       event_id, severity, acquisition_date, copy to output_path
        raise NotImplementedError

    def write_geojson(self, flood_polygons_path: Path, report: FloodReport) -> Path:
        """
        Enrich the raw flood polygons GeoJSON with per-feature attributes
        from the flood report (severity, area, event metadata).

        Parameters
        ----------
        flood_polygons_path : Path
            Raw GeoJSON from FloodPredictor.vectorize_mask().
        report : FloodReport
            Parsed flood report.

        Returns
        -------
        Path
            Path to the enriched GeoJSON.
        """
        output_path = self.output_dir / f"{report.event_id}_flood_extent.geojson"
        # TODO: load GeoJSON with geopandas, add attribute columns from report,
        #       save to output_path
        raise NotImplementedError
