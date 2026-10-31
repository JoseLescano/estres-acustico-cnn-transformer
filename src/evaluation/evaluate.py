"""
Evaluación del modelo híbrido CNN-Transformer sobre el conjunto de prueba,
con comparación contra un método estadístico lineal tradicional
(objetivo específico 4 del proyecto de tesis).

El baseline tradicional NO usa deep learning: extrae estadísticos simples
del espectrograma (media, desviación estándar, máximo, energía total por
banda de frecuencia) y ajusta una Regresión Lineal (mínimos cuadrados)
sobre esas features — representa el enfoque "clásico" frente al cual se
justifica la complejidad añadida de la arquitectura híbrida.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LinearRegression
from torch.utils.data import DataLoader

from src.config import DEVICE, MODELS_SAVED_DIR, REPORTS_DIR, BATCH_SIZE
from src.data.load_dataset import build_datasets
from src.models.hybrid_model import HybridCNNTransformer
from src.training.metrics import compute_all_metrics, EvaluationMetrics


def _extract_statistical_features(spectrogram: torch.Tensor) -> np.ndarray:
    """
    Convierte un espectrograma (1, freq_bins, time_frames) en un vector de
    estadísticos simples, usado como entrada del baseline de Regresión Lineal.

    Args:
        spectrogram: tensor (1, freq_bins, time_frames).

    Returns:
        Array 1D de features: [media, std, máximo, mínimo, energía_por_banda...]
    """
    spec = spectrogram.squeeze(0).numpy()  # (freq_bins, time_frames)

    global_stats = np.array(
        [spec.mean(), spec.std(), spec.max(), spec.min()], dtype=np.float32
    )

    # Energía promedio por banda de frecuencia (colapsando el eje temporal),
    # submuestreada a 16 bandas para mantener el baseline "simple" por diseño.
    n_bands = 16
    band_edges = np.linspace(0, spec.shape[0], n_bands + 1, dtype=int)
    band_energy = np.array(
        [
            spec[band_edges[i]:band_edges[i + 1]].mean() if band_edges[i + 1] > band_edges[i] else 0.0
            for i in range(n_bands)
        ],
        dtype=np.float32,
    )

    return np.concatenate([global_stats, band_energy])


def _dataset_to_numpy(dataloader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
    """Convierte un DataLoader completo a arrays NumPy (features, labels)."""
    features_list = []
    labels_list = []

    for spectrograms, labels in dataloader:
        for i in range(spectrograms.size(0)):
            features_list.append(_extract_statistical_features(spectrograms[i]))
            labels_list.append(labels[i].item())

    return np.stack(features_list), np.array(labels_list, dtype=np.float32)


def evaluate_traditional_baseline(
    train_loader: DataLoader, test_loader: DataLoader
) -> EvaluationMetrics:
    """
    Entrena y evalúa el baseline de Regresión Lineal sobre features
    estadísticas simples del espectrograma.

    Returns:
        EvaluationMetrics del baseline sobre el conjunto de prueba.
    """
    X_train, y_train = _dataset_to_numpy(train_loader)
    X_test, y_test = _dataset_to_numpy(test_loader)

    baseline = LinearRegression()
    baseline.fit(X_train, y_train)

    y_pred = baseline.predict(X_test)

    return compute_all_metrics(
        torch.from_numpy(y_pred.astype(np.float32)),
        torch.from_numpy(y_test),
    )


def evaluate_hybrid_model(
    model: HybridCNNTransformer, test_loader: DataLoader
) -> EvaluationMetrics:
    """
    Evalúa el modelo híbrido CNN-Transformer ya entrenado sobre el
    conjunto de prueba.

    Returns:
        EvaluationMetrics del modelo híbrido sobre el conjunto de prueba.
    """
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for spectrograms, targets in test_loader:
            spectrograms = spectrograms.to(DEVICE)
            predictions = model(spectrograms).cpu()

            all_preds.append(predictions)
            all_targets.append(targets)

    return compute_all_metrics(torch.cat(all_preds), torch.cat(all_targets))


def compare_models(
    checkpoint_path: Path | None = None,
    report_path: Path | None = None,
) -> dict:
    """
    Compara el modelo híbrido CNN-Transformer contra el baseline de
    Regresión Lineal sobre el mismo conjunto de prueba, y guarda un
    reporte JSON con ambos resultados.

    Args:
        checkpoint_path: ruta al checkpoint del modelo híbrido entrenado
            (ver src/training/train.py). Por defecto: models_saved/best_model.pt
        report_path: ruta donde guardar el reporte JSON de comparación.
            Por defecto: reports/comparison_report.json

    Returns:
        Diccionario con las métricas de ambos modelos y la mejora relativa.

    Raises:
        FileNotFoundError: si no existe un checkpoint entrenado.
    """
    checkpoint_path = checkpoint_path or (MODELS_SAVED_DIR / "best_model.pt")
    report_path = report_path or (REPORTS_DIR / "comparison_report.json")

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"No se encontró un checkpoint entrenado en {checkpoint_path}. "
            "Ejecuta primero src/training/train.py (Fase 4)."
        )

    train_dataset, _val_dataset, test_dataset = build_datasets()
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # --- Modelo híbrido ---
    model = HybridCNNTransformer().to(DEVICE)
    # Warmup para materializar nn.LazyLinear antes de cargar el state_dict
    with torch.no_grad():
        sample_batch, _ = next(iter(test_loader))
        model(sample_batch.to(DEVICE))

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])

    hybrid_metrics = evaluate_hybrid_model(model, test_loader)

    # --- Baseline tradicional ---
    baseline_metrics = evaluate_traditional_baseline(train_loader, test_loader)

    # --- Comparación ---
    def _relative_improvement(hybrid_value: float, baseline_value: float) -> float:
        if baseline_value == 0:
            return 0.0
        return (baseline_value - hybrid_value) / abs(baseline_value)

    report = {
        "hybrid_model": hybrid_metrics.to_dict(),
        "traditional_baseline": baseline_metrics.to_dict(),
        "relative_improvement": {
            "rmse": _relative_improvement(hybrid_metrics.rmse, baseline_metrics.rmse),
            "mae": _relative_improvement(hybrid_metrics.mae, baseline_metrics.mae),
            "r2_gain": hybrid_metrics.r2 - baseline_metrics.r2,
        },
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("=== Modelo híbrido CNN-Transformer ===")
    print(hybrid_metrics)
    print("\n=== Baseline: Regresión Lineal (features estadísticas) ===")
    print(baseline_metrics)
    print(f"\nReporte guardado en: {report_path}")

    return report


if __name__ == "__main__":
    compare_models()
