"""
Loop de entrenamiento del modelo híbrido CNN-Transformer.

Incluye:
    - Forward de "calentamiento" antes de crear el optimizador (requerido
      por el uso de nn.LazyLinear en HybridCNNTransformer.input_projection).
    - Entrenamiento con early stopping sobre la pérdida de validación.
    - Checkpointing del mejor modelo en models_saved/.
    - Logging de métricas a TensorBoard (reports/../runs, vía SummaryWriter).

Uso:
    python -m src.training.train
"""

from __future__ import annotations

import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.config import (
    BATCH_SIZE,
    LEARNING_RATE,
    NUM_EPOCHS,
    WEIGHT_DECAY,
    DEVICE,
    MODELS_SAVED_DIR,
    RANDOM_SEED,
)
from src.data.load_dataset import build_datasets
from src.models.hybrid_model import HybridCNNTransformer
from src.training.metrics import compute_all_metrics, EvaluationMetrics


class EarlyStopping:
    """
    Detiene el entrenamiento si la pérdida de validación no mejora
    durante `patience` épocas consecutivas, para evitar sobreajuste
    y desperdiciar cómputo (relevante dado el presupuesto $0 en GPU gratuita).
    """

    def __init__(self, patience: int = 8, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0

    def step(self, val_loss: float) -> bool:
        """
        Returns:
            True si el entrenamiento debe detenerse.
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False

        self.counter += 1
        return self.counter >= self.patience


def _warmup_forward_pass(model: HybridCNNTransformer, dataloader: DataLoader) -> None:
    """
    Ejecuta un forward con un solo batch para materializar los parámetros
    de nn.LazyLinear ANTES de construir el optimizador. Sin este paso, el
    optimizador no vería esos parámetros y nunca se actualizarían.
    """
    model.eval()
    with torch.no_grad():
        sample_batch, _ = next(iter(dataloader))
        model(sample_batch.to(DEVICE))
    model.train()


def _run_epoch(
    model: HybridCNNTransformer,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
) -> tuple[float, EvaluationMetrics]:
    """
    Corre una época completa. Si `optimizer` es None, corre en modo
    evaluación (sin backward ni actualización de pesos).

    Returns:
        (pérdida_promedio, métricas_agregadas)
    """
    is_training = optimizer is not None
    model.train() if is_training else model.eval()

    total_loss = 0.0
    n_samples = 0
    all_preds = []
    all_targets = []

    context = torch.enable_grad() if is_training else torch.no_grad()

    with context:
        for spectrograms, targets in dataloader:
            spectrograms = spectrograms.to(DEVICE)
            targets = targets.to(DEVICE)

            predictions = model(spectrograms)
            loss = criterion(predictions, targets)

            if is_training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            batch_size = targets.size(0)
            total_loss += loss.item() * batch_size
            n_samples += batch_size

            all_preds.append(predictions.detach().cpu())
            all_targets.append(targets.detach().cpu())

    avg_loss = total_loss / n_samples if n_samples > 0 else float("nan")
    metrics = compute_all_metrics(torch.cat(all_preds), torch.cat(all_targets))

    return avg_loss, metrics


def train_model(
    num_epochs: int = NUM_EPOCHS,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
    weight_decay: float = WEIGHT_DECAY,
    early_stopping_patience: int = 8,
    checkpoint_path: Path | None = None,
) -> HybridCNNTransformer:
    """
    Entrena el modelo híbrido CNN-Transformer de principio a fin.

    Args:
        num_epochs: número máximo de épocas.
        batch_size: tamaño de batch para train/val/test.
        learning_rate: tasa de aprendizaje del optimizador Adam.
        weight_decay: regularización L2.
        early_stopping_patience: épocas sin mejora antes de detener.
        checkpoint_path: ruta donde guardar el mejor modelo. Por defecto:
            models_saved/best_model.pt

    Returns:
        El modelo entrenado, con los pesos del mejor checkpoint cargados.
    """
    torch.manual_seed(RANDOM_SEED)

    checkpoint_path = checkpoint_path or (MODELS_SAVED_DIR / "best_model.pt")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    train_dataset, val_dataset, _test_dataset = build_datasets()

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = HybridCNNTransformer().to(DEVICE)

    # Paso obligatorio antes de crear el optimizador (ver docstring de _warmup_forward_pass)
    _warmup_forward_pass(model, train_loader)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    criterion = nn.MSELoss()
    early_stopping = EarlyStopping(patience=early_stopping_patience)

    writer = SummaryWriter()
    best_val_loss = float("inf")

    for epoch in range(1, num_epochs + 1):
        start_time = time.time()

        train_loss, train_metrics = _run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_metrics = _run_epoch(model, val_loader, criterion, optimizer=None)

        elapsed = time.time() - start_time

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("RMSE/val", val_metrics.rmse, epoch)
        writer.add_scalar("MAE/val", val_metrics.mae, epoch)
        writer.add_scalar("F1/val", val_metrics.f1, epoch)

        print(
            f"[Época {epoch:03d}/{num_epochs}] "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"val: {val_metrics} | {elapsed:.1f}s"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_metrics": val_metrics.to_dict(),
                },
                checkpoint_path,
            )
            print(f"  -> Nuevo mejor modelo guardado en {checkpoint_path}")

        if early_stopping.step(val_loss):
            print(f"Early stopping activado en la época {epoch}.")
            break

    writer.close()

    # Carga el mejor checkpoint antes de retornar el modelo
    best_checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(best_checkpoint["model_state_dict"])

    return model


if __name__ == "__main__":
    train_model()
