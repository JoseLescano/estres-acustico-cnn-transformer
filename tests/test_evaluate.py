"""
Pruebas del módulo de evaluación: extracción de features estadísticas
y comparación contra el baseline de Regresión Lineal, usando datos
sintéticos en memoria (sin audio real ni checkpoints entrenados).
"""

import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.evaluate import (  # noqa: E402
    _extract_statistical_features,
    _dataset_to_numpy,
    evaluate_traditional_baseline,
)


class _SyntheticSpectrogramDataset(Dataset):
    """Dataset falso para probar el pipeline de evaluación sin audio real."""

    def __init__(self, n_samples: int = 12, freq_bins: int = 32, time_frames: int = 20, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.spectrograms = [
            torch.from_numpy(rng.normal(size=(1, freq_bins, time_frames)).astype(np.float32))
            for _ in range(n_samples)
        ]
        # Etiquetas correlacionadas con la media del espectrograma, para que
        # el baseline de Regresión Lineal tenga algo real que aprender.
        self.labels = [
            torch.tensor(float(np.clip(spec.mean().item() * 0.1 + 0.5, 0, 1)))
            for spec in self.spectrograms
        ]

    def __len__(self) -> int:
        return len(self.spectrograms)

    def __getitem__(self, index: int):
        return self.spectrograms[index], self.labels[index]


def test_extract_statistical_features_shape():
    spectrogram = torch.randn(1, 32, 20)
    features = _extract_statistical_features(spectrogram)

    # 4 estadísticos globales + 16 bandas de frecuencia = 20 features
    assert features.shape == (20,)
    assert features.dtype == np.float32


def test_extract_statistical_features_is_deterministic():
    spectrogram = torch.randn(1, 32, 20)
    features_a = _extract_statistical_features(spectrogram)
    features_b = _extract_statistical_features(spectrogram)

    assert np.array_equal(features_a, features_b)


def test_dataset_to_numpy_matches_dataset_length():
    dataset = _SyntheticSpectrogramDataset(n_samples=10)
    loader = DataLoader(dataset, batch_size=3, shuffle=False)

    X, y = _dataset_to_numpy(loader)

    assert X.shape[0] == 10
    assert y.shape[0] == 10
    assert X.shape[1] == 20  # mismo tamaño de feature vector


def test_evaluate_traditional_baseline_returns_valid_metrics():
    train_dataset = _SyntheticSpectrogramDataset(n_samples=20, seed=1)
    test_dataset = _SyntheticSpectrogramDataset(n_samples=8, seed=2)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    metrics = evaluate_traditional_baseline(train_loader, test_loader)

    assert metrics.mse >= 0
    assert metrics.rmse >= 0
    assert metrics.mae >= 0
