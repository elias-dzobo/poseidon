"""
U-Net inference on full Sentinel-1 scenes.

Handles tiled inference for large rasters that exceed GPU memory,
stitches tiles back together with overlap-blending, and post-processes
the raw probability map into a clean binary flood mask GeoTIFF.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from src.models.unet.architecture import UNet

logger = logging.getLogger(__name__)


class FloodPredictor:
    """
    Runs trained U-Net on a feature stack GeoTIFF and produces a flood mask.

    Parameters
    ----------
    model_path : str | Path
        Path to the saved checkpoint (.pt file).
    tile_size : int
        Inference tile size in pixels. Default: 512.
    overlap : int
        Overlap between adjacent tiles for blending. Default: 64.
    confidence_threshold : float
        Sigmoid probability threshold for flood/no-flood. Default: 0.5.
    device : str, optional
        'cuda', 'mps', or 'cpu'. Auto-detected if None.
    """

    def __init__(
        self,
        model_path: str | Path,
        tile_size: int = 512,
        overlap: int = 64,
        confidence_threshold: float = 0.5,
        device: Optional[str] = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.tile_size = tile_size
        self.overlap = overlap
        self.confidence_threshold = confidence_threshold
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model()

    def predict(
        self,
        feature_path: Path,
        output_dir: Optional[Path] = None,
    ) -> tuple[Path, Path]:
        """
        Run flood prediction on a feature stack GeoTIFF.

        Parameters
        ----------
        feature_path : Path
            7-channel feature stack GeoTIFF.
        output_dir : Path, optional
            Directory to write outputs. Defaults to feature_path parent.

        Returns
        -------
        tuple[Path, Path]
            - probability_map_path: float32 GeoTIFF, values in [0, 1]
            - flood_mask_path: binary uint8 GeoTIFF, values in {0, 1}
        """
        logger.info(f"Running flood prediction on: {feature_path.name}")
        features, profile = self._load_features(feature_path)
        prob_map = self._predict_tiled(features)
        flood_mask = (prob_map >= self.confidence_threshold).astype(np.uint8)
        flood_mask = self._post_process(flood_mask)

        out_dir = output_dir or feature_path.parent
        prob_path = self._save_raster(prob_map, profile, out_dir, feature_path.stem + "_prob")
        mask_path = self._save_raster(flood_mask, profile, out_dir, feature_path.stem + "_mask")
        logger.info(f"Flood mask saved: {mask_path}")
        return prob_path, mask_path

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_model(self) -> UNet:
        """Load U-Net from checkpoint."""
        checkpoint = torch.load(self.model_path, map_location=self.device)
        model = UNet()
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.device)
        model.eval()
        logger.info(f"U-Net loaded from: {self.model_path}")
        return model

    def _load_features(self, feature_path: Path) -> tuple[np.ndarray, dict]:
        """
        Load feature stack as numpy array + rasterio profile.

        Returns
        -------
        tuple[np.ndarray, dict]
            - features: (C, H, W) float32 array
            - profile: rasterio metadata dict
        """
        # TODO: use rasterio to read all bands and return (array, profile)
        raise NotImplementedError

    def _predict_tiled(self, features: np.ndarray) -> np.ndarray:
        """
        Slide a tile window across the full scene and average overlapping predictions.

        Uses a Gaussian weight map for smooth blending at tile boundaries.

        Parameters
        ----------
        features : np.ndarray
            (C, H, W) float32 feature array.

        Returns
        -------
        np.ndarray
            (H, W) float32 probability map.
        """
        # TODO: implement sliding window with overlap, Gaussian blending,
        #       batched GPU inference
        raise NotImplementedError

    def _post_process(self, mask: np.ndarray) -> np.ndarray:
        """
        Post-process raw binary mask:
          1. Morphological closing — fill small holes within flood patches
          2. Remove small isolated patches below min_area threshold
          3. Smooth boundaries

        Parameters
        ----------
        mask : np.ndarray
            Raw binary mask (H, W) uint8.

        Returns
        -------
        np.ndarray
            Cleaned binary mask.
        """
        # TODO: scipy.ndimage morphological ops + connected component filtering
        raise NotImplementedError

    def _save_raster(
        self,
        array: np.ndarray,
        profile: dict,
        output_dir: Path,
        name: str,
    ) -> Path:
        """Write a numpy array as a GeoTIFF preserving CRS and transform."""
        # TODO: update profile dtype, write with rasterio
        raise NotImplementedError

    def vectorize_mask(self, mask_path: Path, output_dir: Optional[Path] = None) -> Path:
        """
        Convert binary flood mask GeoTIFF to GeoJSON polygons.

        Parameters
        ----------
        mask_path : Path
            Binary flood mask GeoTIFF.
        output_dir : Path, optional
            Output directory. Defaults to mask_path parent.

        Returns
        -------
        Path
            Path to the GeoJSON file with flood polygons.
        """
        # TODO: rasterio.features.shapes → shapely polygons → geopandas → GeoJSON
        raise NotImplementedError
