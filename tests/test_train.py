"""
Prueba de integración del loop de entrenamiento: verifica que el ciclo
completo (warmup -> optimizer -> _run_epoch con backward -> checkpoint)
funcione de punta a punta, usando un dataset sintético en memoria en vez
de audio real y del CSV de metadatos.
"""

import sys
from pathlib import Path

import torch
from torch.nn.parameter import UninitializedParameter
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.hybrid_model import HybridCNNTransformer  # noqa: E402
from src.models.transformer_block import TransformerBlock  # noqa: E402
from src.training.train import _warmup_forward_pass, _run_epoch, EarlyStopping  # noqa: E402


class _SyntheticSpectrogramDataset(Dataset):
    """Dataset falso: espectrogramas y etiquetas aleatorias, solo para probar el loop."""

    def __init__(self, n_samples: int = 8, freq_bins: int = 64, time_frames: int = 40):
        self.n_samples = n_samples
        self.freq_bins = freq_bins
        self.time_frames = time_frames

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, index: int):
        spectrogram = torch.randn(1, self.freq_bins, self.time_frames)
        label = torch.rand(1).squeeze()
        return spectrogram, label


def test_warmup_materializes_lazy_linear_parameters():
    model = HybridCNNTransformer(d_model=16)
    model.transformer = TransformerBlock(d_model=16, n_heads=2, n_layers=1, dim_feedforward=32)

    dataset = _SyntheticSpectrogramDataset()
    loader = DataLoader(dataset, batch_size=4)

    # Filtrar únicamente los parámetros que ya están inicializados antes del warmup
    params_before = sum(
        p.numel() for p in model.parameters() if not isinstance(p, UninitializedParameter)
    )

    _warmup_forward_pass(model, loader)

    params_after = sum(p.numel() for p in model.parameters())

    assert params_after > params_before  # input_projection ya se materializó


def test_run_epoch_training_updates_weights():
    model = HybridCNNTransformer(d_model=16)
    model.transformer = TransformerBlock(d_model=16, n_heads=2, n_layers=1, dim_feedforward=32)

    dataset = _SyntheticSpectrogramDataset()
    loader = DataLoader(dataset, batch_size=4)

    _warmup_forward_pass(model, loader)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.MSELoss()

    weights_before = model.regression_head[0].weight.clone()
    _run_epoch(model, loader, criterion, optimizer)
    weights_after = model.regression_head[0].weight

    assert not torch.allclose(weights_before, weights_after)


def test_run_epoch_eval_mode_does_not_update_weights():
    model = HybridCNNTransformer(d_model=16)
    model.transformer = TransformerBlock(d_model=16, n_heads=2, n_layers=1, dim_feedforward=32)

    dataset = _SyntheticSpectrogramDataset()
    loader = DataLoader(dataset, batch_size=4)

    _warmup_forward_pass(model, loader)

    weights_before = model.regression_head[0].weight.clone()
    _run_epoch(model, loader, torch.nn.MSELoss(), optimizer=None)
    weights_after = model.regression_head[0].weight

    assert torch.allclose(weights_before, weights_after)


def test_early_stopping_triggers_after_patience_exceeded():
    early_stopping = EarlyStopping(patience=3, min_delta=1e-4)

    assert early_stopping.step(1.0) is False   # mejora inicial
    assert early_stopping.step(1.0) is False   # sin mejora, counter=1
    assert early_stopping.step(1.0) is False   # sin mejora, counter=2
    assert early_stopping.step(1.0) is True    # sin mejora, counter=3 >= patience


def test_early_stopping_resets_on_improvement():
    early_stopping = EarlyStopping(patience=2, min_delta=1e-4)

    early_stopping.step(1.0)
    early_stopping.step(1.0)          # counter=1
    assert early_stopping.step(0.5) is False  # mejora -> counter vuelve a 0
    assert early_stopping.counter == 0