"""
Carga y preparación del dataset de audio para el modelo CNN-Transformer.

Se espera un archivo de metadatos CSV en `data/raw/metadata.csv` con,
al menos, las columnas:

    filename,stress_index
    ate_zona1_001.wav,0.72
    ate_zona1_002.wav,0.35
    ...

Donde `filename` es relativo a `data/raw/` y `stress_index` es el valor
objetivo (índice de estrés acústico, normalizado en [0, 1] o en la escala
que se defina en el instrumento de medición).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset, random_split

from src.config import (
    RAW_DATA_DIR,
    SAMPLE_RATE,
    AUDIO_DURATION_SECONDS,
    N_FFT,
    HOP_LENGTH,
    GABOR_SIGMA,
    TRAIN_SPLIT,
    VAL_SPLIT,
    TEST_SPLIT,
    RANDOM_SEED,
)
from src.data.gabor_spectrogram import (
    compute_gabor_spectrogram,
    normalize_spectrogram,
    pad_or_trim_spectrogram,
)


@dataclass(frozen=True)
class AudioRecord:
    """Referencia a un archivo de audio y su etiqueta asociada."""
    filepath: Path
    stress_index: float


def load_metadata(metadata_csv: Path) -> List[AudioRecord]:
    """
    Lee el archivo de metadatos y valida su contenido.

    Args:
        metadata_csv: ruta al archivo CSV de metadatos.

    Returns:
        Lista de AudioRecord con la ruta absoluta de cada audio y su etiqueta.

    Raises:
        FileNotFoundError: si el CSV o alguno de los audios referenciados no existe.
        ValueError: si el CSV no tiene las columnas requeridas o hay valores inválidos.
    """
    if not metadata_csv.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo de metadatos en: {metadata_csv}. "
            "Crea un CSV con columnas 'filename,stress_index' dentro de data/raw/."
        )

    records: List[AudioRecord] = []

    with open(metadata_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        required_columns = {"filename", "stress_index"}
        if reader.fieldnames is None or not required_columns.issubset(set(reader.fieldnames)):
            raise ValueError(
                f"El CSV de metadatos debe tener las columnas {required_columns}, "
                f"se encontraron: {reader.fieldnames}"
            )

        for row_number, row in enumerate(reader, start=2):
            filename = row["filename"].strip()
            raw_label = row["stress_index"].strip()

            try:
                stress_index = float(raw_label)
            except ValueError as exc:
                raise ValueError(
                    f"Valor de stress_index inválido en la fila {row_number}: '{raw_label}'"
                ) from exc

            filepath = RAW_DATA_DIR / filename
            if not filepath.exists():
                raise FileNotFoundError(
                    f"Archivo de audio referenciado en la fila {row_number} no existe: {filepath}"
                )

            records.append(AudioRecord(filepath=filepath, stress_index=stress_index))

    if not records:
        raise ValueError(f"El archivo de metadatos {metadata_csv} no contiene registros.")

    return records


def _load_and_segment_audio(filepath: Path) -> np.ndarray:
    """
    Carga un archivo de audio y lo ajusta a una duración fija mediante
    recorte o relleno con silencio (ceros).

    Args:
        filepath: ruta al archivo .wav.

    Returns:
        Señal de audio mono como array 1D de float32, de longitud fija
        (SAMPLE_RATE * AUDIO_DURATION_SECONDS muestras).
    """
    audio, sr = sf.read(str(filepath), dtype="float32", always_2d=False)

    if audio.ndim > 1:
        # Mezclar canales a mono si el archivo viene estéreo
        audio = audio.mean(axis=1)

    if sr != SAMPLE_RATE:
        raise ValueError(
            f"El archivo {filepath} tiene sample rate {sr} Hz, se esperaba {SAMPLE_RATE} Hz. "
            "Reconvierte el audio antes de agregarlo al dataset."
        )

    target_length = int(SAMPLE_RATE * AUDIO_DURATION_SECONDS)

    if len(audio) >= target_length:
        return audio[:target_length]

    pad_width = target_length - len(audio)
    return np.pad(audio, (0, pad_width), mode="constant", constant_values=0.0)


class AcousticStressDataset(Dataset):
    """
    Dataset de PyTorch que entrega pares (espectrograma_de_gabor, stress_index)
    a partir de archivos de audio y su archivo de metadatos.
    """

    def __init__(self, records: List[AudioRecord], target_frames: int | None = None):
        """
        Args:
            records: lista de AudioRecord ya validados (ver `load_metadata`).
            target_frames: número fijo de frames temporales del espectrograma.
                Si es None, se calcula automáticamente a partir de
                AUDIO_DURATION_SECONDS, N_FFT y HOP_LENGTH.
        """
        self.records = records
        self.target_frames = target_frames or self._compute_expected_frames()

    @staticmethod
    def _compute_expected_frames() -> int:
        n_samples = int(SAMPLE_RATE * AUDIO_DURATION_SECONDS)
        return 1 + (n_samples - N_FFT) // HOP_LENGTH

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        record = self.records[index]

        audio = _load_and_segment_audio(record.filepath)

        spectrogram = compute_gabor_spectrogram(
            audio,
            sample_rate=SAMPLE_RATE,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            sigma=GABOR_SIGMA,
        )
        spectrogram = normalize_spectrogram(spectrogram)
        spectrogram = pad_or_trim_spectrogram(spectrogram, self.target_frames)

        # Shape final: (canales=1, n_freq_bins, n_time_frames) — listo para Conv2D
        spectrogram_tensor = torch.from_numpy(spectrogram).unsqueeze(0).float()
        label_tensor = torch.tensor(record.stress_index, dtype=torch.float32)

        return spectrogram_tensor, label_tensor


def build_datasets(
    metadata_csv: Path | None = None,
) -> Tuple[AcousticStressDataset, AcousticStressDataset, AcousticStressDataset]:
    """
    Construye los splits de entrenamiento, validación y prueba a partir
    del archivo de metadatos, usando las proporciones definidas en config.py.

    Args:
        metadata_csv: ruta al CSV de metadatos. Por defecto: data/raw/metadata.csv.

    Returns:
        Tupla (train_dataset, val_dataset, test_dataset).

    Raises:
        ValueError: si las proporciones de split no suman 1.0.
    """
    metadata_csv = metadata_csv or (RAW_DATA_DIR / "metadata.csv")

    total_split = TRAIN_SPLIT + VAL_SPLIT + TEST_SPLIT
    if abs(total_split - 1.0) > 1e-9:
        raise ValueError(
            f"Las proporciones de split deben sumar 1.0, suman {total_split}. "
            "Revisa TRAIN_SPLIT, VAL_SPLIT y TEST_SPLIT en src/config.py."
        )

    records = load_metadata(metadata_csv)
    full_dataset = AcousticStressDataset(records)

    n_total = len(full_dataset)
    n_train = int(n_total * TRAIN_SPLIT)
    n_val = int(n_total * VAL_SPLIT)
    n_test = n_total - n_train - n_val  # el resto, para no perder muestras por redondeo

    generator = torch.Generator().manual_seed(RANDOM_SEED)
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [n_train, n_val, n_test], generator=generator
    )

    return train_dataset, val_dataset, test_dataset
