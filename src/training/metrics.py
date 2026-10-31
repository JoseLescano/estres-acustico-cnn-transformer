"""
Métricas de evaluación para la tarea de regresión del índice de estrés acústico.

Todas las funciones aceptan tensores de PyTorch (predicción y objetivo) con
la misma forma (batch,) y devuelven un float de Python, listas para loguear
en TensorBoard o en la bitácora de resultados (reports/results.md).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import torch


def mean_squared_error(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """MSE: penaliza fuertemente los errores grandes (cuadrático)."""
    return torch.mean((y_pred - y_true) ** 2).item()


def root_mean_squared_error(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """RMSE: en las mismas unidades que el índice de estrés acústico."""
    return torch.sqrt(torch.mean((y_pred - y_true) ** 2)).item()


def mean_absolute_error(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """MAE: error promedio absoluto, más robusto a outliers que el MSE."""
    return torch.mean(torch.abs(y_pred - y_true)).item()


def r2_score(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """
    Coeficiente de determinación R².
    1.0 = predicción perfecta, 0.0 = equivalente a predecir siempre la media,
    puede ser negativo si el modelo es peor que esa media.
    """
    ss_res = torch.sum((y_true - y_pred) ** 2)
    ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)

    if ss_tot.item() < 1e-12:
        # Todos los valores objetivo son iguales: R² no está bien definido.
        return 0.0

    return (1.0 - ss_res / ss_tot).item()


def f1_score_binary(y_pred: torch.Tensor, y_true: torch.Tensor, threshold: float = 0.5) -> float:
    """
    F1-Score binarizando la predicción y el objetivo con un umbral.
    Útil si además de la regresión se quiere reportar la capacidad del
    modelo de distinguir "estrés acústico alto" vs "bajo" (ver objetivo
    específico 3 del proyecto de tesis).

    Args:
        y_pred: predicciones continuas del modelo.
        y_true: valores objetivo continuos.
        threshold: umbral de binarización (mismo para ambos).
    """
    pred_binary = (y_pred >= threshold).int()
    true_binary = (y_true >= threshold).int()

    true_positives = torch.sum((pred_binary == 1) & (true_binary == 1)).item()
    false_positives = torch.sum((pred_binary == 1) & (true_binary == 0)).item()
    false_negatives = torch.sum((pred_binary == 0) & (true_binary == 1)).item()

    precision_denom = true_positives + false_positives
    recall_denom = true_positives + false_negatives

    precision = true_positives / precision_denom if precision_denom > 0 else 0.0
    recall = true_positives / recall_denom if recall_denom > 0 else 0.0

    if precision + recall == 0:
        return 0.0

    return 2 * (precision * recall) / (precision + recall)


@dataclass
class EvaluationMetrics:
    """Contenedor de todas las métricas calculadas para un conjunto de datos."""

    mse: float
    rmse: float
    mae: float
    r2: float
    f1: float

    def to_dict(self) -> dict:
        return asdict(self)

    def __str__(self) -> str:
        return (
            f"MSE={self.mse:.4f} | RMSE={self.rmse:.4f} | "
            f"MAE={self.mae:.4f} | R2={self.r2:.4f} | F1={self.f1:.4f}"
        )


def compute_all_metrics(
    y_pred: torch.Tensor, y_true: torch.Tensor, f1_threshold: float = 0.5
) -> EvaluationMetrics:
    """
    Calcula el conjunto completo de métricas definidas para el proyecto
    (ver objetivo específico 3 del proyecto de tesis: MSE, RMSE, MAE, F1-Score).

    Args:
        y_pred: predicciones del modelo, tensor (N,).
        y_true: valores reales, tensor (N,).
        f1_threshold: umbral de binarización para el F1-Score.

    Returns:
        EvaluationMetrics con los 5 valores calculados.
    """
    return EvaluationMetrics(
        mse=mean_squared_error(y_pred, y_true),
        rmse=root_mean_squared_error(y_pred, y_true),
        mae=mean_absolute_error(y_pred, y_true),
        r2=r2_score(y_pred, y_true),
        f1=f1_score_binary(y_pred, y_true, threshold=f1_threshold),
    )
