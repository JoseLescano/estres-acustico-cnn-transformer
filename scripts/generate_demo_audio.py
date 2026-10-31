"""Genera WAV sintéticos reproducibles para probar el pipeline local."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import AUDIO_DURATION_SECONDS, RAW_DATA_DIR, SAMPLE_RATE


def _normalize(signal: np.ndarray) -> np.ndarray:
    peak = np.max(np.abs(signal))
    return (signal / peak * 0.85).astype(np.float32) if peak else signal.astype(np.float32)


def generate_demo_audio(output_dir: Path = RAW_DATA_DIR) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    samples = int(SAMPLE_RATE * AUDIO_DURATION_SECONDS)
    time = np.arange(samples, dtype=np.float32) / SAMPLE_RATE

    signals = {}
    labels = {}
    for variant in range(1, 11):
        frequency = 160 + variant * 20
        filename = f"demo_low_stress_{variant}.wav"
        signals[filename] = (0.025 + variant * 0.001) * np.sin(
            2 * np.pi * frequency * time
        )
        labels[filename] = 0.10 + (variant - 1) * 0.01

    for variant in range(1, 11):
        filename = f"demo_high_stress_{variant}.wav"
        signals[filename] = (0.25 + variant * 0.01) * rng.normal(size=samples)
        labels[filename] = 0.70 + (variant - 1) * 0.02

    transient_start = int(1.8 * SAMPLE_RATE)
    transient_end = int(2.2 * SAMPLE_RATE)
    for variant in range(1, 11):
        frequency = 700 + variant * 100
        filename = f"demo_transient_{variant}.wav"
        signal = (0.025 + variant * 0.002) * rng.normal(size=samples)
        signal[transient_start:transient_end] += 0.7 * np.sin(
            2 * np.pi * frequency * time[transient_start:transient_end]
        )
        signals[filename] = signal
        labels[filename] = 0.40 + (variant - 1) * 0.025
    for filename, signal in signals.items():
        sf.write(str(output_dir / filename), _normalize(signal), SAMPLE_RATE, subtype="PCM_16")

    with open(output_dir / "metadata.csv", "w", encoding="utf-8", newline="") as metadata:
        writer = csv.DictWriter(metadata, fieldnames=["filename", "stress_index"])
        writer.writeheader()
        for filename, label in labels.items():
            writer.writerow({"filename": filename, "stress_index": label})


if __name__ == "__main__":
    generate_demo_audio()
    print(f"Archivos de prueba generados en: {RAW_DATA_DIR}")
