"""
Bloque Transformer: modela dependencias temporales de largo alcance sobre
la secuencia de características extraídas por la CNN.

A diferencia de una RNN/LSTM, la self-attention permite que cualquier frame
temporal "atienda" directamente a cualquier otro, sin degradación por
distancia — relevante para el ruido no estacionario, donde un evento
transitorio al inicio de la ventana puede correlacionar con otro al final.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from src.config import (
    TRANSFORMER_D_MODEL,
    TRANSFORMER_N_HEADS,
    TRANSFORMER_N_LAYERS,
    TRANSFORMER_DIM_FEEDFORWARD,
    TRANSFORMER_DROPOUT,
)


class PositionalEncoding(nn.Module):
    """
    Codificación posicional senoidal fija (Vaswani et al., 2017).
    Necesaria porque la self-attention, a diferencia de una RNN, no tiene
    noción intrínseca del orden temporal de los frames.
    """

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # No es un parámetro entrenable, pero debe moverse junto al modelo (to(device))
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor (batch, seq_len, d_model).

        Returns:
            tensor (batch, seq_len, d_model), con la codificación posicional sumada.
        """
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


class TransformerBlock(nn.Module):
    """
    Encoder Transformer configurado a partir de los hiperparámetros de config.py.
    """

    def __init__(
        self,
        d_model: int = TRANSFORMER_D_MODEL,
        n_heads: int = TRANSFORMER_N_HEADS,
        n_layers: int = TRANSFORMER_N_LAYERS,
        dim_feedforward: int = TRANSFORMER_DIM_FEEDFORWARD,
        dropout: float = TRANSFORMER_DROPOUT,
    ):
        super().__init__()

        if d_model % n_heads != 0:
            raise ValueError(
                f"TRANSFORMER_D_MODEL ({d_model}) debe ser divisible entre "
                f"TRANSFORMER_N_HEADS ({n_heads})."
            )

        self.positional_encoding = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor (batch, seq_len, d_model).

        Returns:
            tensor (batch, seq_len, d_model), contextualizado con self-attention.
        """
        x = self.positional_encoding(x)
        return self.encoder(x)
