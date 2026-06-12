"""
Dataset loader for U-Net flood segmentation training.

Supports two data sources:
  1. Sen1Floods11 — Berkeley benchmark dataset (446 hand-labeled chips)
     https://github.com/cloudtostreet/Sen1Floods11
  2. Ghana local labels — Copernicus EMS / UNOSAT flood maps for Accra events

Each sample is a (feature_stack, flood_mask) pair:
  - feature_stack: (7, 512, 512) float32 tensor
  - flood_mask:    (1, 512, 512) float32 tensor  {0: no flood, 1: flood}
"""

from __future__ import annotations

import random
import logging
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class FloodDataset(Dataset):
    """
    PyTorch Dataset for flood segmentation.

    Loads pre-built feature stacks and corresponding flood masks.
    Applies optional data augmentation.

    Parameters
    ----------
    feature_dir : str | Path
        Directory containing feature stack GeoTIFFs (one per event/date).
    mask_dir : str | Path
        Directory containing flood mask GeoTIFFs (matching filenames).
    tile_size : int
        Spatial size of tiles to sample from each scene. Default: 512.
    augment : bool
        Whether to apply random spatial augmentations. Default: False.
    transform : callable, optional
        Additional torchvision transforms applied to the feature tensor.
    """

    def __init__(
        self,
        feature_dir: str | Path,
        mask_dir: str | Path,
        tile_size: int = 512,
        augment: bool = False,
        transform: Optional[Callable] = None,
    ) -> None:
        self.feature_dir = Path(feature_dir)
        self.mask_dir = Path(mask_dir)
        self.tile_size = tile_size
        self.augment = augment
        self.transform = transform

        self.samples = self._discover_samples()
        logger.info(
            f"FloodDataset: {len(self.samples)} samples from {self.feature_dir}"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        feature_path, mask_path = self.samples[idx]

        features = self._load_feature_stack(feature_path)  # (C, H, W)
        mask = self._load_mask(mask_path)                   # (1, H, W)

        features, mask = self._sample_tile(features, mask)

        if self.augment:
            features, mask = self._augment(features, mask)

        if self.transform:
            features = self.transform(features)

        return features, mask

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _discover_samples(self) -> list[tuple[Path, Path]]:
        """
        Match feature stacks to their corresponding flood masks.

        Assumes matching filenames between feature_dir and mask_dir.

        Returns
        -------
        list[tuple[Path, Path]]
            List of (feature_path, mask_path) pairs.
        """
        # TODO: glob feature_dir for *.tif, find matching mask in mask_dir
        raise NotImplementedError

    def _load_feature_stack(self, path: Path) -> torch.Tensor:
        """
        Load a multi-band feature GeoTIFF as a float32 tensor.

        Returns
        -------
        torch.Tensor
            Shape (C, H, W), float32, values in [0, 1].
        """
        # TODO: use rasterio to read all bands, convert to torch tensor
        raise NotImplementedError

    def _load_mask(self, path: Path) -> torch.Tensor:
        """
        Load a binary flood mask GeoTIFF.

        Returns
        -------
        torch.Tensor
            Shape (1, H, W), float32, values in {0.0, 1.0}.
        """
        # TODO: read single band, binarize, return float32 tensor
        raise NotImplementedError

    def _sample_tile(
        self, features: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Randomly sample a tile_size x tile_size crop from the scene.

        Biases sampling toward tiles containing flood pixels to address
        class imbalance (most pixels are not flooded).
        """
        # TODO: random crop, with optional flood-biased sampling
        raise NotImplementedError

    @staticmethod
    def _augment(
        features: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply random spatial augmentations consistently to features and mask.

        Augmentations:
          - Random horizontal flip
          - Random vertical flip
          - Random 90-degree rotation
        """
        # TODO: apply same random transforms to features and mask
        if random.random() > 0.5:
            features = torch.flip(features, dims=[2])
            mask = torch.flip(mask, dims=[2])
        if random.random() > 0.5:
            features = torch.flip(features, dims=[1])
            mask = torch.flip(mask, dims=[1])
        k = random.randint(0, 3)
        features = torch.rot90(features, k=k, dims=[1, 2])
        mask = torch.rot90(mask, k=k, dims=[1, 2])
        return features, mask


class Sen1Floods11Dataset(FloodDataset):
    """
    Specialised loader for the Sen1Floods11 benchmark dataset.

    Downloads and parses the official dataset structure:
      data/
        S1Hand/    — Sentinel-1 chips
        LabelHand/ — hand-labeled flood masks

    Parameters
    ----------
    root_dir : str | Path
        Root directory of the Sen1Floods11 download.
    split : str
        One of 'train', 'val', 'test'.
    """

    SPLITS_URL = (
        "https://raw.githubusercontent.com/cloudtostreet/Sen1Floods11/"
        "master/splits/"
    )

    def __init__(
        self,
        root_dir: str | Path,
        split: str = "train",
        **kwargs,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.split = split
        self._split_files = self._load_split_file()
        super().__init__(
            feature_dir=self.root_dir / "S1Hand",
            mask_dir=self.root_dir / "LabelHand",
            **kwargs,
        )

    def _load_split_file(self) -> list[str]:
        """
        Load the official train/val/test split CSV from Sen1Floods11.

        Returns list of scene IDs for this split.
        """
        # TODO: download split CSV from SPLITS_URL or load from local cache
        raise NotImplementedError

    def _discover_samples(self) -> list[tuple[Path, Path]]:
        """Override to filter samples by split file."""
        # TODO: filter discovered samples to only those in self._split_files
        raise NotImplementedError
