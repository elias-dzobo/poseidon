"""
Dataset for LFM2.5 VL LoRA fine-tuning.

Each training sample is an (image, prompt, report_json) triple:
  - image:       SAR false-color composite PNG (with flood mask overlay)
  - prompt:      Instruction + metadata context (user turn)
  - report_json: Ground-truth flood assessment JSON (assistant turn)

Ground-truth reports are sourced from:
  - Copernicus EMS rapid mapping products (Ghana flood events)
  - UNOSAT flood analysis maps
  - Synthetically generated reports from verified U-Net masks
    (for events with no official EMS/UNOSAT annotation)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable, Optional

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class FloodReportDataset(Dataset):
    """
    Dataset of (composite_image, prompt, report_json) triples for VLM fine-tuning.

    Directory structure expected:
      training_pairs/
        images/          — composite PNG files
        reports/         — matching JSON report files (same stem)
        metadata.json    — per-image metadata (date, region, season, sources)

    Parameters
    ----------
    data_dir : str | Path
        Root of the training_pairs directory.
    processor : transformers.AutoProcessor
        VLM processor for tokenising text + encoding image.
    max_seq_length : int
        Maximum token length for combined prompt + response.
    split : str
        'train', 'val', or 'test'.
    split_ratios : tuple[float, float, float]
        Train/val/test split ratios. Default: (0.75, 0.15, 0.10).
    """

    def __init__(
        self,
        data_dir: str | Path,
        processor,
        max_seq_length: int = 1024,
        split: str = "train",
        split_ratios: tuple[float, float, float] = (0.75, 0.15, 0.10),
    ) -> None:
        self.data_dir = Path(data_dir)
        self.processor = processor
        self.max_seq_length = max_seq_length
        self.split = split

        self.metadata = self._load_metadata()
        self.samples = self._split_samples(split_ratios)
        logger.info(
            f"FloodReportDataset [{split}]: {len(self.samples)} samples"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.samples[idx]
        image = self._load_image(sample["image_path"])
        prompt = self._build_prompt(sample["metadata"])
        response = self._load_report_json(sample["report_path"])
        return self._encode(image, prompt, response)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_metadata(self) -> dict:
        """Load the per-image metadata.json file."""
        metadata_path = self.data_dir / "metadata.json"
        if not metadata_path.exists():
            logger.warning(f"metadata.json not found at {metadata_path}")
            return {}
        with open(metadata_path) as f:
            return json.load(f)

    def _split_samples(
        self, ratios: tuple[float, float, float]
    ) -> list[dict]:
        """
        Discover all image-report pairs and split by ratio.

        Returns list of dicts: {image_path, report_path, metadata}.
        """
        # TODO: glob images dir, match reports, apply deterministic train/val/test split
        raise NotImplementedError

    def _load_image(self, image_path: Path):
        """Load a composite PNG as a PIL Image."""
        # TODO: PIL.Image.open(image_path).convert("RGB")
        raise NotImplementedError

    def _load_report_json(self, report_path: Path) -> str:
        """Load a JSON report and return as a formatted string (assistant response)."""
        with open(report_path) as f:
            data = json.load(f)
        return json.dumps(data, indent=2)

    def _build_prompt(self, meta: dict) -> str:
        """Build the user-turn instruction prompt from sample metadata."""
        return (
            f"Acquisition date: {meta.get('acquisition_date', 'unknown')}\n"
            f"Region: {meta.get('region', {})}\n"
            f"Season: {meta.get('season', 'unknown')}\n"
            f"Data sources: {', '.join(meta.get('data_sources', []))}\n\n"
            "Analyze the satellite image and generate the flood assessment JSON report."
        )

    def _encode(self, image, prompt: str, response: str) -> dict[str, torch.Tensor]:
        """
        Encode image + prompt + response through the VLM processor.

        For causal LM fine-tuning, the input_ids include both prompt and
        response tokens; labels mask out the prompt tokens (only supervise
        on the assistant response).

        Returns
        -------
        dict
            HuggingFace-compatible batch dict with input_ids, attention_mask, labels.
        """
        # TODO:
        # 1. Build chat-format messages list
        # 2. processor(messages, images=image, return_tensors='pt', truncation=True,
        #              max_length=max_seq_length)
        # 3. Build labels: copy input_ids, set prompt token positions to -100
        raise NotImplementedError
