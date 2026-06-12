"""
Sentinel-2 optical preprocessing pipeline.

Handles:
  - Band extraction (B03 Green, B08 NIR, B11 SWIR1, B12 SWIR2)
  - Cloud and cloud-shadow masking (s2cloudless)
  - Atmospheric correction validation (L2A already corrected)
  - Water index computation: NDWI, MNDWI
  - Resampling to 10m resolution
  - Clipping to AOI
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class OpticalPreprocessor:
    """
    Preprocessing pipeline for Sentinel-2 L2A scenes.

    Parameters
    ----------
    output_dir : str | Path
        Directory to write preprocessed outputs.
    target_crs : str
        Output CRS (default: EPSG:32630 — UTM 30N).
    target_resolution : int
        Output spatial resolution in metres (default: 10).
    """

    # Sentinel-2 band indices used in this pipeline
    BANDS = {
        "B03": "green",    # 10m — for NDWI
        "B08": "nir",      # 10m — for NDWI
        "B11": "swir1",    # 20m — for MNDWI (resampled to 10m)
        "B12": "swir2",    # 20m — auxiliary
        "SCL": "scene_classification",  # 20m — for cloud masking
    }

    def __init__(
        self,
        output_dir: str | Path = "data/processed/sentinel2",
        target_crs: str = "EPSG:32630",
        target_resolution: int = 10,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.target_crs = target_crs
        self.target_resolution = target_resolution

    def process(self, scene_path: Path, bbox: Optional[list[float]] = None) -> Path:
        """
        Run the full preprocessing chain on a raw Sentinel-2 .zip scene.

        Parameters
        ----------
        scene_path : Path
            Path to the raw .zip or .SAFE directory.
        bbox : list[float], optional
            [min_lat, min_lon, max_lat, max_lon] to clip output.

        Returns
        -------
        Path
            Path to the preprocessed multi-band GeoTIFF
            (bands: Green, NIR, SWIR1, NDWI, MNDWI, cloud_mask).
        """
        logger.info(f"Preprocessing optical scene: {scene_path.name}")
        bands = self._extract_bands(scene_path)
        cloud_mask = self._build_cloud_mask(bands["SCL"])
        bands = self._resample_bands(bands)
        if bbox:
            bands = self._clip_to_bbox(bands, bbox)
        ndwi = self.compute_ndwi(bands["B03"], bands["B08"])
        mndwi = self.compute_mndwi(bands["B03"], bands["B11"])
        output_path = self._export(bands, ndwi, mndwi, cloud_mask, scene_path)
        logger.info(f"Optical preprocessing complete: {output_path}")
        return output_path

    # ------------------------------------------------------------------
    # Private pipeline steps
    # ------------------------------------------------------------------

    def _extract_bands(self, scene_path: Path) -> dict[str, np.ndarray]:
        """
        Extract required bands from the Sentinel-2 .SAFE directory structure.

        Returns a dict mapping band name → numpy array.
        """
        # TODO: parse .SAFE directory, locate JP2 files for each band,
        #       read with rasterio, return as dict
        raise NotImplementedError

    def _build_cloud_mask(self, scl: np.ndarray) -> np.ndarray:
        """
        Build a binary cloud + cloud-shadow mask from the SCL band.

        Sentinel-2 SCL classes masked out:
          3  = cloud shadow
          8  = medium probability cloud
          9  = high probability cloud
          10 = thin cirrus
          11 = snow/ice

        Returns
        -------
        np.ndarray
            Binary mask: 1 = valid pixel, 0 = cloud / shadow / snow.
        """
        # TODO: mask SCL classes 3, 8, 9, 10, 11
        raise NotImplementedError

    def _resample_bands(self, bands: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """
        Resample all bands to the target resolution (10m).

        B11 and B12 are native 20m — upsample with bilinear interpolation.
        SCL is 20m — upsample with nearest-neighbour to preserve class values.
        """
        # TODO: use rasterio reproject/resample
        raise NotImplementedError

    def _clip_to_bbox(
        self, bands: dict[str, np.ndarray], bbox: list[float]
    ) -> dict[str, np.ndarray]:
        """Clip all bands to the AOI bounding box."""
        # TODO: use rasterio.mask.mask with bbox polygon
        raise NotImplementedError

    def _export(
        self,
        bands: dict[str, np.ndarray],
        ndwi: np.ndarray,
        mndwi: np.ndarray,
        cloud_mask: np.ndarray,
        source_scene: Path,
    ) -> Path:
        """
        Write all bands + indices + cloud mask to a single multi-band GeoTIFF.

        Band order: Green, NIR, SWIR1, NDWI, MNDWI, cloud_mask.
        """
        # TODO: use rasterio to write stacked GeoTIFF with correct CRS and transform
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Water index computations
    # ------------------------------------------------------------------

    @staticmethod
    def compute_ndwi(green: np.ndarray, nir: np.ndarray) -> np.ndarray:
        """
        Normalized Difference Water Index (McFeeters 1996).

        NDWI = (Green - NIR) / (Green + NIR)

        Positive values indicate water. Threshold typically 0.0–0.3.
        Sensitive to built-up areas (false positives in urban Accra).
        """
        with np.errstate(divide="ignore", invalid="ignore"):
            ndwi = np.where(
                (green + nir) > 0,
                (green.astype(float) - nir) / (green.astype(float) + nir),
                np.nan,
            )
        return ndwi.astype(np.float32)

    @staticmethod
    def compute_mndwi(green: np.ndarray, swir1: np.ndarray) -> np.ndarray:
        """
        Modified NDWI (Xu 2006).

        MNDWI = (Green - SWIR1) / (Green + SWIR1)

        Preferred over NDWI in urban areas (like central Accra) because
        SWIR better suppresses built-up land signal.
        """
        with np.errstate(divide="ignore", invalid="ignore"):
            mndwi = np.where(
                (green + swir1) > 0,
                (green.astype(float) - swir1) / (green.astype(float) + swir1),
                np.nan,
            )
        return mndwi.astype(np.float32)
