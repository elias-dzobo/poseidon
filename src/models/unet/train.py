"""
U-Net training loop.

Trains the flood segmentation model on Sen1Floods11 + Ghana local labels.
Supports mixed-precision training, early stopping, and metric logging.

Loss: Combined BCE + Dice loss (configurable weights).
Metrics: IoU (Intersection over Union), F1 score, precision, recall.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from src.models.unet.architecture import UNet
from src.models.unet.dataset import FloodDataset

logger = logging.getLogger(__name__)


class DiceLoss(nn.Module):
    """
    Soft Dice loss for binary segmentation.

    Directly optimises the Dice coefficient — better suited than BCE
    for imbalanced classes (flood pixels are rare).
    """

    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)
        intersection = (probs_flat * targets_flat).sum()
        dice = (2.0 * intersection + self.smooth) / (
            probs_flat.sum() + targets_flat.sum() + self.smooth
        )
        return 1.0 - dice


class CombinedLoss(nn.Module):
    """
    Weighted combination of BCE and Dice losses.

    L = bce_weight * BCE + dice_weight * Dice
    """

    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        pos_weight: Optional[float] = None,
    ) -> None:
        super().__init__()
        pw = torch.tensor([pos_weight]) if pos_weight else None
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pw)
        self.dice = DiceLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return (
            self.bce_weight * self.bce(logits, targets)
            + self.dice_weight * self.dice(logits, targets)
        )


class SegmentationMetrics:
    """Accumulates and computes IoU, F1, precision, recall over an epoch."""

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold
        self.reset()

    def reset(self) -> None:
        self.tp = self.fp = self.fn = self.tn = 0

    def update(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        preds = (torch.sigmoid(logits) >= self.threshold).float()
        t = targets.float()
        self.tp += (preds * t).sum().item()
        self.fp += (preds * (1 - t)).sum().item()
        self.fn += ((1 - preds) * t).sum().item()
        self.tn += ((1 - preds) * (1 - t)).sum().item()

    @property
    def iou(self) -> float:
        denom = self.tp + self.fp + self.fn
        return self.tp / denom if denom > 0 else 0.0

    @property
    def f1(self) -> float:
        denom = 2 * self.tp + self.fp + self.fn
        return 2 * self.tp / denom if denom > 0 else 0.0

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom > 0 else 0.0

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom > 0 else 0.0

    def summary(self) -> dict[str, float]:
        return {
            "iou": self.iou,
            "f1": self.f1,
            "precision": self.precision,
            "recall": self.recall,
        }


class UNetTrainer:
    """
    Training orchestrator for the flood segmentation U-Net.

    Parameters
    ----------
    model : UNet
        U-Net model instance.
    train_dataset : FloodDataset
        Training data.
    val_dataset : FloodDataset
        Validation data.
    config : dict
        Training hyperparameters (from model_config.yaml unet.training section).
    checkpoint_dir : str | Path
        Directory to save model checkpoints.
    device : str
        'cuda', 'mps', or 'cpu'.
    """

    def __init__(
        self,
        model: UNet,
        train_dataset: FloodDataset,
        val_dataset: FloodDataset,
        config: dict,
        checkpoint_dir: str | Path = "models/weights/unet",
        device: Optional[str] = None,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.config = config
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=config["batch_size"],
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

        self.criterion = CombinedLoss(
            bce_weight=config.get("bce_weight", 0.5),
            dice_weight=config.get("dice_weight", 0.5),
            pos_weight=config.get("pos_weight", 3.0),
        )
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config["learning_rate"],
            weight_decay=config.get("weight_decay", 1e-5),
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config["epochs"]
        )
        self.scaler = torch.cuda.amp.GradScaler(enabled=(self.device == "cuda"))

        self._best_val_iou = 0.0
        self._patience_counter = 0

    def train(self) -> None:
        """Run the full training loop."""
        epochs = self.config["epochs"]
        patience = self.config.get("early_stopping_patience", 10)

        for epoch in range(1, epochs + 1):
            train_loss = self._train_epoch()
            val_metrics = self._val_epoch()

            logger.info(
                f"Epoch {epoch}/{epochs} | "
                f"train_loss={train_loss:.4f} | "
                f"val_iou={val_metrics['iou']:.4f} | "
                f"val_f1={val_metrics['f1']:.4f}"
            )

            self.scheduler.step()

            # Checkpoint best model
            if val_metrics["iou"] > self._best_val_iou:
                self._best_val_iou = val_metrics["iou"]
                self._patience_counter = 0
                self._save_checkpoint(epoch, val_metrics)
            else:
                self._patience_counter += 1
                if self._patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break

    def _train_epoch(self) -> float:
        """One training epoch. Returns mean loss."""
        self.model.train()
        total_loss = 0.0
        for features, masks in self.train_loader:
            features = features.to(self.device)
            masks = masks.to(self.device)
            self.optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=(self.device == "cuda")):
                logits = self.model(features)
                loss = self.criterion(logits, masks)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            total_loss += loss.item()
        return total_loss / len(self.train_loader)

    def _val_epoch(self) -> dict[str, float]:
        """One validation epoch. Returns metric dict."""
        self.model.eval()
        metrics = SegmentationMetrics()
        with torch.no_grad():
            for features, masks in self.val_loader:
                features = features.to(self.device)
                masks = masks.to(self.device)
                logits = self.model(features)
                metrics.update(logits, masks)
        return metrics.summary()

    def _save_checkpoint(self, epoch: int, metrics: dict) -> None:
        """Save model weights + training state."""
        path = self.checkpoint_dir / f"unet_epoch{epoch:03d}_iou{metrics['iou']:.4f}.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "metrics": metrics,
            },
            path,
        )
        logger.info(f"Checkpoint saved: {path}")
