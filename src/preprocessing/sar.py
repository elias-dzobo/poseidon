"""
SAR preprocessing pipeline.

Applies the standard preprocessing chain to raw Sentinel-1 GRD scenes:
  1. Apply orbit file
  2. Thermal noise removal
  3. Radiometric calibration (DN → sigma0 in dB)
  4. Speckle filtering (Lee filter)
  5. Range-Doppler terrain correction (using SRTM DEM)
  6. Subsetting to AOI
  7. Stack VV + VH bands into a single GeoTIFF

Uses SNAP/snappy (Python interface to ESA SNAP) or pyroSAR as backend.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class SARPreprocessor:
    """
    Full preprocessing chain for Sentinel-1 GRD scenes.

    Parameters
    ----------
    dem_path : Path
        Path to the DEM GeoTIFF used for terrain correction.
    output_dir : str | Path
        Directory to write preprocessed outputs.
    filter_size : int
        Lee speckle filter window size (default: 7).
    target_crs : str
        Output coordinate reference system (default: EPSG:32630 — UTM 30N).
    """

    def __init__(
        self,
        dem_path: Optional[Path] = None,
        output_dir: str | Path = "data/processed/sentinel1",
        filter_size: int = 7,
        target_crs: str = "EPSG:32630",
    ) -> None:
        self.dem_path = dem_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.filter_size = filter_size
        self.target_crs = target_crs

    def process(self, scene_path: Path, bbox: Optional[list[float]] = None) -> Path:
        """
        Run the full preprocessing chain on a raw Sentinel-1 .zip scene.

        Parameters
        ----------
        scene_path : Path
            Path to the raw .zip or .SAFE file.
        bbox : list[float], optional
            [min_lat, min_lon, max_lat, max_lon] to clip output.

        Returns
        -------
        Path
            Path to the preprocessed dual-pol GeoTIFF (VV, VH bands).
        """
        logger.info(f"Preprocessing SAR scene: {scene_path.name}")
        scene_path = self._apply_orbit_file(scene_path)
        scene_path = self._remove_thermal_noise(scene_path)
        scene_path = self._calibrate(scene_path)
        scene_path = self._apply_speckle_filter(scene_path)
        scene_path = self._terrain_correct(scene_path)
        if bbox:
            scene_path = self._subset(scene_path, bbox)
        output_path = self._export(scene_path)
        logger.info(f"SAR preprocessing complete: {output_path}")
        return output_path

    # ------------------------------------------------------------------
    # Private pipeline steps
    # ------------------------------------------------------------------

    def _apply_orbit_file(self, scene_path: Path) -> Path:
        """Apply precise orbit file to correct satellite position metadata."""
        # TODO: use SNAP Apply-Orbit-File operator
        raise NotImplementedError

    def _remove_thermal_noise(self, scene_path: Path) -> Path:
        """Remove thermal noise from raw GRD bands."""
        # TODO: use SNAP ThermalNoiseRemoval operator
        raise NotImplementedError

    def _calibrate(self, scene_path: Path) -> Path:
        """
        Radiometric calibration: convert DN values to sigma0 backscatter (dB).

        sigma0_dB = 10 * log10(sigma0_linear)
        """
        # TODO: use SNAP Calibration operator, output sigma0 in dB
        raise NotImplementedError

    def _apply_speckle_filter(self, scene_path: Path) -> Path:
        """
        Apply Lee speckle filter to reduce multiplicative noise.

        Lee filter is preferred for flood detection as it preserves
        the sharp boundaries between water and land.
        """
        # TODO: use SNAP Speckle-Filter operator with Lee, filter_size x filter_size
        raise NotImplementedError

    def _terrain_correct(self, scene_path: Path) -> Path:
        """
        Range-Doppler terrain correction to map sigma0 to geographic coordinates.

        Uses the SRTM DEM to account for topographic distortion.
        """
        # TODO: use SNAP Range-Doppler-Terrain-Correction operator
        #       with self.dem_path and self.target_crs
        raise NotImplementedError

    def _subset(self, scene_path: Path, bbox: list[float]) -> Path:
        """Clip the scene to the AOI bounding box."""
        # TODO: use SNAP Subset operator with WKT polygon from bbox
        raise NotImplementedError

    def _export(self, scene_path: Path) -> Path:
        """Export the processed SNAP product to GeoTIFF."""
        # TODO: use SNAP Write operator, format GeoTIFF-BigTIFF
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def compute_polarisation_ratio(vv: np.ndarray, vh: np.ndarray) -> np.ndarray:
        """
        Compute VV/VH polarisation ratio.

        Water surfaces show low backscatter in both polarisations with a
        characteristic VV/VH ratio distinct from vegetation and urban areas.

        Parameters
        ----------
        vv : np.ndarray
            VV backscatter in linear scale.
        vh : np.ndarray
            VH backscatter in linear scale.

        Returns
        -------
        np.ndarray
            VV/VH ratio (linear scale).
        """
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(vh > 0, vv / vh, np.nan)
        return ratio

    @staticmethod
    def db_to_linear(db: np.ndarray) -> np.ndarray:
        """Convert backscatter from dB to linear scale."""
        return 10.0 ** (db / 10.0)

    @staticmethod
    def linear_to_db(linear: np.ndarray) -> np.ndarray:
        """Convert backscatter from linear to dB scale."""
        with np.errstate(divide="ignore"):
            return 10.0 * np.log10(linear)
