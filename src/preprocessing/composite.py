"""
SAR false-color RGB composite generator.

Converts dual-polarisation SAR data into an RGB image that can be
ingested by the LFM2.5 VL vision encoder (which expects natural RGB input).

False-color encoding:
  R = VV backscatter (sigma0, dB normalised to [0, 255])
  G = VH backscatter (sigma0, dB normalised to [0, 255])
  B = VV/VH polarisation ratio (normalised to [0, 255])

Water surfaces appear dark / near-black due to specular reflection
(low VV, low VH, characteristic ratio). Urban areas appear bright.
Vegetation shows high VH relative to VV.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class SARCompositeGenerator:
    """
    Generates false-color RGB composites from preprocessed Sentinel-1 GeoTIFFs.

    These composites are used as visual input to the LFM2.5 VL model,
    optionally overlaid with the U-Net flood mask.

    Parameters
    ----------
    output_dir : str | Path
        Directory to write composite PNG / GeoTIFF images.
    vv_range : tuple[float, float]
        (min_dB, max_dB) for VV normalisation. Default: (-25, 0).
    vh_range : tuple[float, float]
        (min_dB, max_dB) for VH normalisation. Default: (-30, -5).
    ratio_range : tuple[float, float]
        (min, max) linear ratio for normalisation. Default: (0.5, 10).
    """

    def __init__(
        self,
        output_dir: str | Path = "data/processed/composites",
        vv_range: tuple[float, float] = (-25.0, 0.0),
        vh_range: tuple[float, float] = (-30.0, -5.0),
        ratio_range: tuple[float, float] = (0.5, 10.0),
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.vv_range = vv_range
        self.vh_range = vh_range
        self.ratio_range = ratio_range

    def generate(
        self,
        sar_path: Path,
        flood_mask_path: Optional[Path] = None,
        output_name: Optional[str] = None,
    ) -> Path:
        """
        Generate a false-color RGB composite, optionally overlaid with flood mask.

        Parameters
        ----------
        sar_path : Path
            Preprocessed Sentinel-1 GeoTIFF (VV band 1, VH band 2).
        flood_mask_path : Path, optional
            Binary flood mask GeoTIFF from U-Net. If provided, flooded pixels
            are highlighted in blue on the composite.
        output_name : str, optional
            Output filename stem. Defaults to sar_path stem + '_composite'.

        Returns
        -------
        Path
            Path to the output composite PNG.
        """
        logger.info(f"Generating SAR composite for: {sar_path.name}")

        vv, vh = self._load_sar_bands(sar_path)
        ratio = self._compute_ratio(vv, vh)

        r = self._normalise_channel(vv, *self.vv_range)
        g = self._normalise_channel(vh, *self.vh_range)
        b = self._normalise_channel(ratio, *self.ratio_range)

        rgb = np.stack([r, g, b], axis=-1)  # (H, W, 3) uint8

        if flood_mask_path is not None:
            rgb = self._overlay_flood_mask(rgb, flood_mask_path)

        stem = output_name or (sar_path.stem + "_composite")
        output_path = self.output_dir / f"{stem}.png"
        self._save_png(rgb, output_path)
        logger.info(f"Composite saved: {output_path}")
        return output_path

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_sar_bands(self, sar_path: Path) -> tuple[np.ndarray, np.ndarray]:
        """Load VV (band 1) and VH (band 2) from the preprocessed SAR GeoTIFF."""
        # TODO: read with rasterio, return (vv, vh) as float32 arrays
        raise NotImplementedError

    def _compute_ratio(self, vv: np.ndarray, vh: np.ndarray) -> np.ndarray:
        """
        Compute VV/VH ratio in linear scale.

        Input VV/VH are in dB — convert to linear first, then divide.
        """
        # TODO: 10**(vv/10) / 10**(vh/10), handle divide-by-zero
        raise NotImplementedError

    @staticmethod
    def _normalise_channel(
        array: np.ndarray, min_val: float, max_val: float
    ) -> np.ndarray:
        """
        Clip and normalise a float array to uint8 [0, 255].

        Parameters
        ----------
        array : np.ndarray
            Input float array.
        min_val, max_val : float
            Physical range to map to [0, 255].

        Returns
        -------
        np.ndarray
            uint8 array.
        """
        clipped = np.clip(array, min_val, max_val)
        normalised = (clipped - min_val) / (max_val - min_val)
        return (normalised * 255).astype(np.uint8)

    def _overlay_flood_mask(
        self, rgb: np.ndarray, flood_mask_path: Path
    ) -> np.ndarray:
        """
        Highlight flooded pixels on the RGB composite.

        Flooded pixels (mask == 1) are tinted blue with 60% opacity
        to make them visually distinct to the VLM vision encoder.

        Parameters
        ----------
        rgb : np.ndarray
            (H, W, 3) uint8 composite.
        flood_mask_path : Path
            Binary GeoTIFF flood mask.

        Returns
        -------
        np.ndarray
            (H, W, 3) uint8 composite with flood overlay.
        """
        # TODO: load mask, blend flooded pixels: rgb[mask==1] = 0.4*rgb + 0.6*(0,0,255)
        raise NotImplementedError

    @staticmethod
    def _save_png(rgb: np.ndarray, output_path: Path) -> None:
        """Save a (H, W, 3) uint8 array as PNG."""
        # TODO: use PIL.Image or rasterio to write PNG
        raise NotImplementedError
