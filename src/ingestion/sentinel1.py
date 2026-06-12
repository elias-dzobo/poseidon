"""
Sentinel-1 SAR data downloader.

Downloads GRD IW dual-polarisation (VV+VH) scenes from the Copernicus
Data Space Ecosystem (CDSE) for a given AOI and date range.

Search uses the CDSE OData v4 catalogue API:
  https://catalogue.dataspace.copernicus.eu/odata/v1/Products

Download uses the CDSE zipper service:
  https://zipper.dataspace.copernicus.eu/odata/v1/Products({id})/$value

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


class Sentinel1Downloader(CDSEDownloaderBase):
    """
    Downloads Sentinel-1 GRD IW scenes from CDSE.

    Parameters
    ----------
    username : str, optional
        CDSE account email. Falls back to CDSE_USERNAME env var.
    password : str, optional
        CDSE account password. Falls back to CDSE_PASSWORD env var.
    output_dir : str | Path
        Directory where raw .zip scenes will be saved.
        Default: data/raw/sentinel1
    """

    def __init__(
        self,
        username: Optional[str] = None,
        password: Optional[str] = None,
        output_dir: str | Path = "data/raw/sentinel1",
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
        polarisation: str = "VV VH",
        product_type: str = "GRD",
        instrument_mode: str = "IW",
        max_results: int = 100,
    ) -> list[dict]:
        """
        Search the CDSE catalogue for Sentinel-1 scenes.

        Parameters
        ----------
        bbox : list[float]
            [min_lat, min_lon, max_lat, max_lon]
        start_date : str | datetime
            Search window start (YYYY-MM-DD or ISO datetime).
        end_date : str | datetime
            Search window end.
        polarisation : str
            Polarisation mode. "VV VH" for dual-pol IW GRD.
        product_type : str
            "GRD" (ground-range detected) or "SLC" (single-look complex).
        instrument_mode : str
            "IW" — Interferometric Wide Swath (standard for flood detection).
        max_results : int
            Maximum number of scenes to return.

        Returns
        -------
        list[dict]
            Sorted by acquisition date (newest first). Each dict has:
              id, name, size, date, footprint, polarisation, orbit_direction.
        """
        wkt = self._bbox_to_wkt(bbox)
        start = self._format_date(start_date)
        end = self._format_date(end_date)

        # Build OData $filter expression
        filters = [
            "Collection/Name eq 'SENTINEL-1'",
            f"OData.CSC.Intersects(area=geography'SRID=4326;{wkt}')",
            f"ContentDate/Start gt {start}",
            f"ContentDate/Start lt {end}",
            (
                "Attributes/OData.CSC.StringAttribute/any("
                "att:att/Name eq 'productType' "
                f"and att/OData.CSC.StringAttribute/Value eq '{product_type}')"
            ),
            (
                "Attributes/OData.CSC.StringAttribute/any("
                "att:att/Name eq 'operationalMode' "
                f"and att/OData.CSC.StringAttribute/Value eq '{instrument_mode}')"
            ),
        ]

        if polarisation:
            filters.append(
                "Attributes/OData.CSC.StringAttribute/any("
                "att:att/Name eq 'polarisationChannels' "
                f"and att/OData.CSC.StringAttribute/Value eq '{polarisation}')"
            )

        odata_filter = " and ".join(filters)

        params = {
            "$filter": odata_filter,
            "$expand": "Attributes",
        }

        logger.info(
            f"Searching Sentinel-1 {product_type}/{instrument_mode} "
            f"{polarisation} | {start_date} → {end_date}"
        )
        scenes = self._paginated_search(params, max_results)

        # Enrich with S1-specific attributes
        for scene in scenes:
            raw = scene.pop("raw", {})
            scene["polarisation"] = self._extract_attr(raw, "polarisationChannels")
            scene["orbit_direction"] = self._extract_attr(raw, "orbitDirection")
            scene["relative_orbit"] = self._extract_attr(raw, "relativeOrbitNumber")
            scene["instrument_mode"] = self._extract_attr(raw, "operationalMode")

        logger.info(f"Sentinel-1 search: {len(scenes)} scenes found")
        return scenes

    def search_pre_post(
        self,
        bbox: list[float],
        flood_date: str | datetime,
        pre_window_days: int = 30,
        post_window_days: int = 5,
        **search_kwargs,
    ) -> tuple[list[dict], list[dict]]:
        """
        Convenience method: find pre-flood and post-flood scene pairs.

        Searches for scenes in a window before the flood event (baseline)
        and immediately after (flood peak).

        Parameters
        ----------
        bbox : list[float]
            [min_lat, min_lon, max_lat, max_lon]
        flood_date : str | datetime
            Approximate date of the flood event.
        pre_window_days : int
            How many days before the flood date to search for baseline scenes.
        post_window_days : int
            How many days after the flood date to search for flood scenes.

        Returns
        -------
        tuple[list[dict], list[dict]]
            (pre_flood_scenes, post_flood_scenes)
        """
        from datetime import timedelta

        if isinstance(flood_date, str):
            flood_date = datetime.fromisoformat(flood_date)

        pre_start = flood_date - timedelta(days=pre_window_days)
        pre_end = flood_date - timedelta(days=1)
        post_start = flood_date
        post_end = flood_date + timedelta(days=post_window_days)

        logger.info(
            f"Searching pre-flood scenes: {pre_start.date()} → {pre_end.date()}"
        )
        pre_scenes = self.search(bbox, pre_start, pre_end, **search_kwargs)

        logger.info(
            f"Searching post-flood scenes: {post_start.date()} → {post_end.date()}"
        )
        post_scenes = self.search(bbox, post_start, post_end, **search_kwargs)

        logger.info(
            f"Pre-flood: {len(pre_scenes)} scene(s), "
            f"post-flood: {len(post_scenes)} scene(s)"
        )
        return pre_scenes, post_scenes
