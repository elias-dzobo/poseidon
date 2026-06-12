"""
Feature engineering and multi-source stack assembly.

Combines preprocessed SAR, optical, and auxiliary layers into a single
aligned multi-channel feature stack ready for U-Net input.

Feature channels (7 total):
  0: VV backscatter (dB)
  1: VH backscatter (dB)
  2: VV/VH ratio (linear)
  3: NDWI
  4: MNDWI
  5: HAND (Height Above Nearest Drainage, metres)
  6: Slope (degrees)
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class FeatureBuilder:
    """
    Assembles aligned multi-channel feature stacks from preprocessed inputs.

    All inputs are resampled/reprojected to match the SAR raster grid
    (10m resolution, EPSG:32630) before stacking.

    Parameters
    ----------
    output_dir : str | Path
        Directory to write feature stack GeoTIFFs.
    """

    FEATURE_NAMES = ["VV", "VH", "VV_VH_ratio", "NDWI", "MNDWI", "HAND", "slope"]
    N_FEATURES = len(FEATURE_NAMES)

    def __init__(self, output_dir: str | Path = "data/features") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def build(
        self,
        sar_path: Path,
        optical_path: Path,
        hand_path: Path,
        slope_path: Path,
        permanent_water_path: Path,
        output_name: str,
    ) -> Path:
        """
        Build a 7-channel feature stack from preprocessed inputs.

        Parameters
        ----------
        sar_path : Path
            Preprocessed Sentinel-1 GeoTIFF (VV, VH bands).
        optical_path : Path
            Preprocessed Sentinel-2 GeoTIFF (includes NDWI, MNDWI bands).
        hand_path : Path
            HAND GeoTIFF.
        slope_path : Path
            Slope GeoTIFF.
        permanent_water_path : Path
            JRC permanent water mask GeoTIFF.
        output_name : str
            Base name for the output file.

        Returns
        -------
        Path
            Path to the 7-channel feature stack GeoTIFF.
        """
        logger.info(f"Building feature stack: {output_name}")

        sar_data = self._load_sar(sar_path)
        optical_data = self._load_optical(optical_path)
        hand = self._load_raster(hand_path)
        slope = self._load_raster(slope_path)

        # Align all layers to SAR grid
        optical_data = self._align_to_reference(optical_data, sar_path)
        hand = self._align_to_reference(hand, sar_path)
        slope = self._align_to_reference(slope, sar_path)

        stack = self._stack(sar_data, optical_data, hand, slope)
        stack = self._apply_permanent_water_mask(stack, permanent_water_path)
        stack = self._normalize(stack)

        output_path = self._export(stack, sar_path, output_name)
        logger.info(f"Feature stack written: {output_path}")
        return output_path

    # ------------------------------------------------------------------
    # Private methods
    # ------------------------------------------------------------------

    def _load_sar(self, sar_path: Path) -> dict[str, np.ndarray]:
        """
        Load VV and VH bands and compute VV/VH ratio.

        Returns dict with keys: vv, vh, ratio.
        """
        # TODO: read with rasterio, compute ratio via SARPreprocessor utility
        raise NotImplementedError

    def _load_optical(self, optical_path: Path) -> dict[str, np.ndarray]:
        """
        Load NDWI and MNDWI from the preprocessed optical GeoTIFF.

        Returns dict with keys: ndwi, mndwi.
        """
        # TODO: read NDWI and MNDWI bands by index from optical GeoTIFF
        raise NotImplementedError

    def _load_raster(self, path: Path) -> np.ndarray:
        """Read a single-band raster as a numpy array."""
        # TODO: use rasterio to read band 1
        raise NotImplementedError

    def _align_to_reference(
        self, data: np.ndarray | dict, reference_path: Path
    ) -> np.ndarray | dict:
        """
        Reproject and resample data to match the reference raster grid.

        Uses bilinear resampling for continuous values (HAND, slope, indices)
        and nearest-neighbour for categorical data.
        """
        # TODO: use rasterio.warp.reproject to match reference CRS, transform, shape
        raise NotImplementedError

    def _stack(
        self,
        sar_data: dict[str, np.ndarray],
        optical_data: dict[str, np.ndarray],
        hand: np.ndarray,
        slope: np.ndarray,
    ) -> np.ndarray:
        """
        Stack all feature channels into a (H, W, C) array.

        Channel order matches FEATURE_NAMES.
        """
        # TODO: np.stack in the order [VV, VH, ratio, NDWI, MNDWI, HAND, slope]
        raise NotImplementedError

    def _apply_permanent_water_mask(
        self, stack: np.ndarray, permanent_water_path: Path
    ) -> np.ndarray:
        """
        Zero-out SAR change signal over permanent water bodies.

        Permanent water pixels are always wet — their backscatter drop
        should not trigger flood alerts.
        """
        # TODO: load mask, set VV/VH delta channels to 0 where mask == 1
        raise NotImplementedError

    def _normalize(self, stack: np.ndarray) -> np.ndarray:
        """
        Per-channel min-max normalization to [0, 1].

        Uses fixed physical ranges where possible:
          VV, VH: [-30, 0] dB
          NDWI, MNDWI: [-1, 1]
          HAND: [0, 30] m (clipped)
          Slope: [0, 45] degrees (clipped)
        """
        # TODO: apply fixed-range normalization per channel
        raise NotImplementedError

    def _export(
        self, stack: np.ndarray, reference_path: Path, output_name: str
    ) -> Path:
        """Write the feature stack as a multi-band GeoTIFF."""
        # TODO: use rasterio, copy CRS/transform from reference_path
        raise NotImplementedError
