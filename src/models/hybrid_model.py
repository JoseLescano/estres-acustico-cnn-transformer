"""
Modelo híbrido CNN-Transformer para la predicción del índice de estrés
acústico a partir de espectrogramas de Gabor.

Pipeline interno:
    espectrograma (batch, 1, freq_bins, frames)
        -> CNNExtractor           -> (batch, C, freq_bins', frames')
        -> proyección a secuencia -> (batch, frames', d_model)
        -> TransformerBlock       -> (batch, frames', d_model)
        -> pooling temporal       -> (batch, d_model)
        -> cabeza de regresión    -> (batch,)  # índice de estrés acústico
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.config import TRANSFORMER_D_MODEL
from src.models.cnn_extractor import CNNExtractor
from src.models.transformer_block import TransformerBlock


class HybridCNNTransformer(nn.Module):
    """
    Modelo híbrido CNN-Transformer con cabeza de regresión.

    NOTA IMPORTANTE: `input_projection` es un nn.LazyLinear, cuyos parámetros
    no se inicializan hasta el primer forward(). En la Fase 4 (entrenamiento),
    el optimizador debe crearse DESPUÉS de un forward de "calentamiento"
    (una pasada con un batch dummy) para que esos parámetros ya existan y el
    optimizador los registre correctamente.
    """

    def __init__(self, d_model: int = TRANSFORMER_D_MODEL):
        super().__init__()

        self.cnn = CNNExtractor()
        self.d_model = d_model

        # Proyecta los canales de salida de la CNN (aplanando el eje de frecuencia)
        # al espacio de embeddings d_model que espera el Transformer.
        # nn.LazyLinear infiere in_features en el primer forward, porque ese
        # valor depende de freq_bins y del número de bloques CNN configurados.
        self.input_projection = nn.LazyLinear(d_model)

        self.transformer = TransformerBlock(d_model=d_model)

        self.regression_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """
        Args:
            spectrogram: tensor (batch, 1, freq_bins, time_frames).

        Returns:
            tensor (batch,) con la predicción del índice de estrés acústico.
        """
        if spectrogram.dim() != 4:
            raise ValueError(
                f"Se esperaba un tensor 4D (batch, 1, freq_bins, time_frames), "
                f"se recibió shape {tuple(spectrogram.shape)}"
            )

        features = self.cnn(spectrogram)  # (batch, C, freq', frames')
        batch_size, channels, freq_dim, time_dim = features.shape

        # Reordena a secuencia temporal: cada frame temporal es un "token"
        # cuyo embedding es la concatenación de canales x frecuencia en ese instante.
        features = features.permute(0, 3, 1, 2)  # (batch, frames', C, freq')
        features = features.reshape(batch_size, time_dim, channels * freq_dim)

        embeddings = self.input_projection(features)  # (batch, frames', d_model)

        contextualized = self.transformer(embeddings)  # (batch, frames', d_model)

        # Pooling temporal por promedio: resume toda la secuencia en un solo vector.
        # Se prefiere sobre tomar solo el último frame porque el estrés acústico
        # depende del comportamiento del ruido a lo largo de toda la ventana, no
        # solo de su instante final.
        pooled = contextualized.mean(dim=1)  # (batch, d_model)

        output = self.regression_head(pooled)  # (batch, 1)
        return output.squeeze(-1)  # (batch,)
