"""
U-Net architecture for pixel-wise flood segmentation.

Standard encoder-decoder with skip connections. Designed to accept
multi-channel satellite feature stacks (7 channels by default).

Reference: Ronneberger et al. (2015) — U-Net: Convolutional Networks
for Biomedical Image Segmentation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Two consecutive (Conv → BN → ReLU) blocks."""

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class EncoderBlock(nn.Module):
    """Downsampling encoder block: MaxPool → DoubleConv."""

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class DecoderBlock(nn.Module):
    """
    Upsampling decoder block: TransposedConv → concat skip → DoubleConv.

    Uses bilinear upsampling + 1x1 conv rather than transposed conv
    to avoid checkerboard artefacts.
    """

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
        )
        self.conv = DoubleConv(in_channels, out_channels, dropout=dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        # Handle odd spatial dimensions
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=True)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """
    U-Net for flood segmentation from multi-channel satellite imagery.

    Parameters
    ----------
    in_channels : int
        Number of input feature channels (default: 7).
        [VV, VH, VV/VH ratio, NDWI, MNDWI, HAND, slope]
    out_channels : int
        Number of output channels. 1 for binary flood/no-flood mask.
    features : list[int]
        Number of filters at each encoder depth level.
        Default: [64, 128, 256, 512]
    dropout : float
        Dropout probability applied in DoubleConv blocks.

    Input
    -----
    x : torch.Tensor
        Shape (B, in_channels, H, W). H and W should be multiples of 32.

    Output
    ------
    torch.Tensor
        Shape (B, 1, H, W). Raw logits — apply sigmoid for probabilities.
    """

    def __init__(
        self,
        in_channels: int = 7,
        out_channels: int = 1,
        features: list[int] = None,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if features is None:
            features = [64, 128, 256, 512]

        # Stem (no pooling)
        self.stem = DoubleConv(in_channels, features[0], dropout=dropout)

        # Encoder
        self.encoders = nn.ModuleList([
            EncoderBlock(features[i], features[i + 1], dropout=dropout)
            for i in range(len(features) - 1)
        ])

        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2, dropout=dropout)

        # Decoder (reverse features)
        decoder_features = [features[-1] * 2] + list(reversed(features))
        self.decoders = nn.ModuleList([
            DecoderBlock(decoder_features[i], decoder_features[i + 1], dropout=dropout)
            for i in range(len(decoder_features) - 1)
        ])

        # Output head
        self.head = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder path — collect skip connections
        skip_connections = []
        x = self.stem(x)
        skip_connections.append(x)

        for encoder in self.encoders:
            x = encoder(x)
            skip_connections.append(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder path — consume skip connections in reverse
        skip_connections = skip_connections[::-1]
        for i, decoder in enumerate(self.decoders):
            x = decoder(x, skip_connections[i])

        return self.head(x)

    def predict_mask(
        self, x: torch.Tensor, threshold: float = 0.5
    ) -> torch.Tensor:
        """
        Run inference and return a binary mask.

        Parameters
        ----------
        x : torch.Tensor
            Input feature stack (B, C, H, W).
        threshold : float
            Sigmoid probability threshold for flood classification.

        Returns
        -------
        torch.Tensor
            Binary mask (B, 1, H, W), dtype torch.bool.
        """
        self.eval()
        with torch.no_grad():
            logits = self(x)
            probs = torch.sigmoid(logits)
            return probs >= threshold
