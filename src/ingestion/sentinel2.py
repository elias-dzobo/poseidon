"""
Sentinel-2 optical data downloader.

Downloads L2A (surface reflectance) scenes from the Copernicus Data Space
Ecosystem (CDSE) for a given AOI and date range. Filters by cloud cover.

Bands used downstream:
  B03 (Green, 10m)  — NDWI numerator
  B08 (NIR, 10m)    — NDWI denominator
  B11 (SWIR1, 20m)  — MNDWI denominator
  B12 (SWIR2, 20m)  — auxiliary
  SCL (20m)         — scene classification for cloud masking

During Ghana's major rainy season (Apr–Jun), expect cloud cover > 80%
for most scenes. SAR is the primary source; Sentinel-2 is opportunistic.

Authentication, streaming download, and retry logic are handled by
CDSEDownloaderBase in auth.py.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.ingestion.auth import CDSEDownloaderBase

logger = logging.getLogger(__name__)


class Sentinel2Downloader(CDSEDownloaderBase):
    """
    Downloads Sentinel-2 L2A scenes from CDSE.

    Parameters
    ----------
    username : str, optional
        CDSE account email. Falls back to CDSE_USERNAME env var.
    password : str, optional
        CDSE account password. Falls back to CDSE_PASSWORD env var.
    output_dir : str | Path
        Directory where raw .zip scenes will be saved.
        Default: data/raw/sentinel2
    """

    def __init__(
        self,
        username: Optional[str] = None,
        password: Optional[str] = None,
        output_dir: str | Path = "data/raw/sentinel2",
    ) -> None:
        super().__init__(username, password, Path(output_dir))

    # ------------------------------------------------------------------
    # Scene search
    # ------------------------------------------------------------------

    def search(
        self,
        bbox: list[float],
        start_date: str | datetime,
        end_date: str | datetime,
        cloud_cover_max: float = 20.0,
        product_type: str = "S2MSI2A",
        max_results: int = 100,
    ) -> list[dict]:
        """
        Search the CDSE catalogue for Sentinel-2 L2A scenes.

        Parameters
        ----------
        bbox : list[float]
            [min_lat, min_lon, max_lat, max_lon]
        start_date : str | datetime
            Search window start (YYYY-MM-DD or ISO datetime).
        end_date : str | datetime
            Search window end.
        cloud_cover_max : float
            Maximum cloud cover percentage to accept (0–100).
            Set higher (e.g. 80) during rainy season when collecting any
            available optical data for partial compositing.
        product_type : str
            "S2MSI2A" = L2A surface reflectance (default, preferred).
            "S2MSI1C" = L1C top-of-atmosphere (fallback if L2A unavailable).
        max_results : int
            Maximum number of scenes to return.

        Returns
        -------
        list[dict]
            Sorted by cloud cover (lowest first). Each dict has:
              id, name, size, date, footprint, cloud_cover, tile_id.
        """
        wkt = self._bbox_to_wkt(bbox)
        start = self._format_date(start_date)
        end = self._format_date(end_date)

        filters = [
            "Collection/Name eq 'SENTINEL-2'",
            f"OData.CSC.Intersects(area=geography'SRID=4326;{wkt}')",
            f"ContentDate/Start gt {start}",
            f"ContentDate/Start lt {end}",
            (
                "Attributes/OData.CSC.StringAttribute/any("
                "att:att/Name eq 'productType' "
                f"and att/OData.CSC.StringAttribute/Value eq '{product_type}')"
            ),
            (
                "Attributes/OData.CSC.DoubleAttribute/any("
                "att:att/Name eq 'cloudCover' "
                f"and att/OData.CSC.DoubleAttribute/Value lt {cloud_cover_max})"
            ),
        ]

        odata_filter = " and ".join(filters)

        params = {
            "$filter": odata_filter,
            "$expand": "Attributes",
        }

        logger.info(
            f"Searching Sentinel-2 {product_type} | "
            f"{start_date} → {end_date} | cloud ≤ {cloud_cover_max}%"
        )
        scenes = self._paginated_search(params, max_results)

        # Enrich with S2-specific attributes
        for scene in scenes:
            raw = scene.pop("raw", {})
            scene["tile_id"] = self._extract_attr(raw, "tileId")
            scene["processing_baseline"] = self._extract_attr(
                raw, "processingBaseline"
            )
            scene["solar_elevation"] = self._extract_attr(raw, "illuminationElevationAngle")

        # Sort by cloud cover ascending (clearest first)
        scenes.sort(key=lambda s: (s.get("cloud_cover") or 100.0))

        logger.info(
            f"Sentinel-2 search: {len(scenes)} scene(s) found "
            f"(cloud cover range: "
            f"{min((s.get('cloud_cover') or 0) for s in scenes):.1f}% – "
            f"{max((s.get('cloud_cover') or 0) for s in scenes):.1f}%)"
            if scenes else "no scenes found"
        )
        return scenes

    def search_dry_season_baseline(
        self,
        bbox: list[float],
        year: int,
        cloud_cover_max: float = 10.0,
    ) -> list[dict]:
        """
        Search for dry-season Sentinel-2 scenes to build a pre-flood baseline.

        Ghana's dry season is roughly December–February. A cloud-free
        dry-season composite gives a clean land/water baseline for
        change detection.

        Parameters
        ----------
        bbox : list[float]
            [min_lat, min_lon, max_lat, max_lon]
        year : int
            Year for the dry season baseline (Dec year-1 → Feb year).
        cloud_cover_max : float
            Maximum acceptable cloud cover. Use a tight threshold (≤10%)
            for clean baseline compositing.

        Returns
        -------
        list[dict]
            Low-cloud-cover scenes from the dry season.
        """
        start = f"{year - 1}-12-01"
        end = f"{year}-02-28"
        logger.info(f"Searching dry-season baseline: {start} → {end}")
        return self.search(
            bbox,
            start_date=start,
            end_date=end,
            cloud_cover_max=cloud_cover_max,
        )
