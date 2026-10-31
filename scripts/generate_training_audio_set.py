"""Genera 100 audios sintéticos de entrenamiento con etiquetas de estrés.

Ejecuta:
    py -3.10 scripts\generate_training_audio_set.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import AUDIO_DURATION_SECONDS, RAW_DATA_DIR, SAMPLE_RATE


def _normalize(signal: np.ndarray) -> np.ndarray:
    peak = np.max(np.abs(signal))
    if peak == 0:
        return signal.astype(np.float32)
    return (signal / peak * 0.85).astype(np.float32)


def generate_training_audio_set(
    count: int = 100,
    output_dir: Path | None = None,
    metadata_csv: Path | None = None,
) -> Path:
    output_dir = output_dir or (RAW_DATA_DIR / "synthetic_training")
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_csv = metadata_csv or (RAW_DATA_DIR / "metadata.csv")
    rng = np.random.default_rng(2026)
    samples = int(SAMPLE_RATE * AUDIO_DURATION_SECONDS)
    time = np.arange(samples, dtype=np.float32) / SAMPLE_RATE

    existing_rows = []
    if metadata_csv.exists():
        with open(metadata_csv, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames and {"filename", "stress_index"}.issubset(set(reader.fieldnames)):
                existing_rows = list(reader)

    seen_names = {row["filename"] for row in existing_rows if row.get("filename")}
    generated_rows = []

    for idx in range(1, count + 1):
        base_freq = 180 + rng.integers(0, 1300)
        noise_amp = 0.04 + rng.random() * 0.26
        tone_amp = 0.10 + rng.random() * 0.30
        panic = rng.random()

        signal = tone_amp * np.sin(2 * np.pi * base_freq * time)
        signal += noise_amp * rng.normal(size=samples)

        burst_start = int((0.8 + rng.random() * 1.8) * SAMPLE_RATE)
        burst_end = min(samples, burst_start + int(0.25 * SAMPLE_RATE))
        burst_freq = 350 + rng.integers(0, 1200)
        burst_amp = 0.18 + rng.random() * 0.55
        signal[burst_start:burst_end] += burst_amp * np.sin(
            2 * np.pi * burst_freq * time[burst_start:burst_end]
        )

        if panic > 0.7:
            amplitude_mod = 1.0 + rng.random() * 1.8
            signal += amplitude_mod * 0.12 * np.sin(2 * np.pi * (40 + rng.integers(0, 110)) * time)

        signal = _normalize(signal)

        filename = f"synthetic_train_{idx:03d}.wav"
        rel_path = f"synthetic_training/{filename}"
        while rel_path in seen_names:
            idx += 1
            filename = f"synthetic_train_{idx:03d}.wav"
            rel_path = f"synthetic_training/{filename}"

        out_path = output_dir / filename
        sf.write(str(out_path), signal, SAMPLE_RATE, subtype="PCM_16")

        stress_value = float(np.clip(0.08 + rng.random() * 0.84 + (0.18 if panic > 0.75 else 0.0), 0.05, 0.95))
        generated_rows.append({"filename": rel_path, "stress_index": f"{stress_value:.4f}"})
        seen_names.add(rel_path)

    if generated_rows:
        with open(metadata_csv, "a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["filename", "stress_index"])
            if metadata_csv.stat().st_size == 0:
                writer.writeheader()
            writer.writerows(generated_rows)

    print(f"Se generaron {len(generated_rows)} archivos WAV en {output_dir}")
    print(f"CSV actualizado: {metadata_csv}")
    return output_dir


if __name__ == "__main__":
    generate_training_audio_set(count=100)
