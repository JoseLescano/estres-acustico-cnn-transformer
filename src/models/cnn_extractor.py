"""
Bloque CNN: extracción de características locales del espectrograma de Gabor.

Actúa como "detector de patrones" tiempo-frecuencia de corto alcance
(bordes de energía, bandas de frecuencia activas, transitorios) antes de
que el Transformer modele las dependencias temporales de largo alcance.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.config import CNN_CHANNELS, CNN_KERNEL_SIZE, CNN_POOL_SIZE


class ConvBlock(nn.Module):
    """Bloque convolucional: Conv2D -> BatchNorm -> ReLU -> MaxPool."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, pool_size: int):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,  # padding "same" para no perder resolución antes del pooling
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=pool_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        return x


class CNNExtractor(nn.Module):
    """
    Extractor CNN de múltiples capas, configurado a partir de CNN_CHANNELS
    (ej. [1, 16, 32, 64] -> 3 bloques convolucionales encadenados).
    """

    def __init__(
        self,
        channels: list[int] | None = None,
        kernel_size: int = CNN_KERNEL_SIZE,
        pool_size: int = CNN_POOL_SIZE,
    ):
        super().__init__()
        channels = channels or CNN_CHANNELS

        if len(channels) < 2:
            raise ValueError(
                f"CNN_CHANNELS debe tener al menos 2 valores (entrada y una capa), "
                f"se recibió: {channels}"
            )

        blocks = []
        for in_ch, out_ch in zip(channels[:-1], channels[1:]):
            blocks.append(ConvBlock(in_ch, out_ch, kernel_size, pool_size))

        self.blocks = nn.Sequential(*blocks)
        self.out_channels = channels[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor (batch, 1, freq_bins, time_frames) — espectrograma de Gabor.

        Returns:
            tensor (batch, out_channels, freq_bins_reducido, time_frames_reducido).
        """
        return self.blocks(x)
