"""
Pruebas del módulo de métricas de evaluación.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.training.metrics import (  # noqa: E402
    mean_squared_error,
    root_mean_squared_error,
    mean_absolute_error,
    r2_score,
    f1_score_binary,
    compute_all_metrics,
)


def test_perfect_prediction_gives_zero_error():
    y_true = torch.tensor([0.1, 0.5, 0.9, 0.3])
    y_pred = y_true.clone()

    assert mean_squared_error(y_pred, y_true) == 0.0
    assert root_mean_squared_error(y_pred, y_true) == 0.0
    assert mean_absolute_error(y_pred, y_true) == 0.0


def test_perfect_prediction_gives_r2_of_one():
    y_true = torch.tensor([0.1, 0.5, 0.9, 0.3, 0.7])
    y_pred = y_true.clone()

    assert abs(r2_score(y_pred, y_true) - 1.0) < 1e-6


def test_r2_handles_constant_target():
    y_true = torch.tensor([0.5, 0.5, 0.5])
    y_pred = torch.tensor([0.4, 0.5, 0.6])

    assert r2_score(y_pred, y_true) == 0.0


def test_rmse_is_sqrt_of_mse():
    y_true = torch.tensor([0.0, 1.0, 0.0, 1.0])
    y_pred = torch.tensor([1.0, 0.0, 1.0, 0.0])

    mse = mean_squared_error(y_pred, y_true)
    rmse = root_mean_squared_error(y_pred, y_true)

    assert abs(rmse - mse ** 0.5) < 1e-6


def test_f1_score_perfect_classification():
    y_true = torch.tensor([0.9, 0.1, 0.8, 0.2])
    y_pred = torch.tensor([0.95, 0.05, 0.85, 0.15])

    f1 = f1_score_binary(y_pred, y_true, threshold=0.5)
    assert abs(f1 - 1.0) < 1e-6


def test_f1_score_worst_case_is_zero():
    y_true = torch.tensor([0.9, 0.9, 0.9])
    y_pred = torch.tensor([0.1, 0.1, 0.1])

    f1 = f1_score_binary(y_pred, y_true, threshold=0.5)
    assert f1 == 0.0


def test_compute_all_metrics_returns_all_fields():
    y_true = torch.tensor([0.2, 0.6, 0.8])
    y_pred = torch.tensor([0.25, 0.55, 0.75])

    metrics = compute_all_metrics(y_pred, y_true)
    result_dict = metrics.to_dict()

    assert set(result_dict.keys()) == {"mse", "rmse", "mae", "r2", "f1"}
    assert all(isinstance(v, float) for v in result_dict.values())
