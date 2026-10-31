"""
Genera la plantilla de metadatos (data/raw/metadata.csv) a partir de los
archivos de audio .wav ya colocados en data/raw/.

El script NO inventa etiquetas: deja la columna `stress_index` vacía para
que el investigador la complete con el valor medido (o calculado a partir
del sonómetro / dataset de referencia) para cada grabación. Sirve además
para validar de antemano que todos los audios cumplan el sample rate
esperado por el pipeline (config.SAMPLE_RATE), evitando descubrir el
problema recién al entrenar.

Uso:
    python -m src.data.generate_metadata_template
    python -m src.data.generate_metadata_template --overwrite
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import soundfile as sf

from src.config import RAW_DATA_DIR, SAMPLE_RATE


def find_audio_files(raw_dir: Path) -> list[Path]:
    """
    Busca recursivamente archivos .wav dentro de `raw_dir`, ignorando
    subcarpetas ocultas.

    Args:
        raw_dir: carpeta raíz donde buscar (normalmente data/raw/).

    Returns:
        Lista de rutas a archivos .wav, ordenada alfabéticamente.
    """
    if not raw_dir.exists():
        raise FileNotFoundError(f"La carpeta {raw_dir} no existe.")

    audio_files = sorted(
        p for p in raw_dir.rglob("*.wav") if not any(part.startswith(".") for part in p.parts)
    )
    return audio_files


def validate_sample_rate(filepath: Path, expected_sr: int) -> tuple[bool, int]:
    """
    Verifica el sample rate de un archivo de audio sin cargarlo completo
    en memoria (usa solo el header).

    Returns:
        (es_valido, sample_rate_real)
    """
    info = sf.info(str(filepath))
    return info.samplerate == expected_sr, info.samplerate


def generate_metadata_template(
    raw_dir: Path | None = None,
    output_csv: Path | None = None,
    overwrite: bool = False,
) -> Path:
    """
    Genera (o actualiza) el CSV de metadatos a partir de los audios encontrados.

    Si el CSV ya existe y `overwrite=False`, conserva las etiquetas ya
    ingresadas para los archivos que sigan presentes, y solo agrega filas
    nuevas para audios que aún no estén registrados — así no se pierde
    trabajo de etiquetado ya hecho.

    Args:
        raw_dir: carpeta con los audios. Por defecto: data/raw/.
        output_csv: ruta de salida del CSV. Por defecto: data/raw/metadata.csv.
        overwrite: si es True, ignora el CSV existente y genera uno nuevo
            desde cero (todas las etiquetas quedan vacías).

    Returns:
        Ruta al CSV generado.
    """
    raw_dir = raw_dir or RAW_DATA_DIR
    output_csv = output_csv or (raw_dir / "metadata.csv")

    audio_files = find_audio_files(raw_dir)

    if not audio_files:
        print(f"No se encontraron archivos .wav en {raw_dir}.")
        print("Coloca tus grabaciones ahí (formato .wav) y vuelve a correr este script.")
        return output_csv

    existing_labels: dict[str, str] = {}
    if output_csv.exists() and not overwrite:
        with open(output_csv, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames and {"filename", "stress_index"}.issubset(set(reader.fieldnames)):
                for row in reader:
                    existing_labels[row["filename"]] = row["stress_index"]

    rows = []
    sample_rate_warnings = []

    for filepath in audio_files:
        relative_name = str(filepath.relative_to(raw_dir))

        is_valid, actual_sr = validate_sample_rate(filepath, SAMPLE_RATE)
        if not is_valid:
            sample_rate_warnings.append((relative_name, actual_sr))

        label = existing_labels.get(relative_name, "")
        rows.append({"filename": relative_name, "stress_index": label})

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "stress_index"])
        writer.writeheader()
        writer.writerows(rows)

    n_labeled = sum(1 for r in rows if r["stress_index"] != "")
    n_total = len(rows)

    print(f"Plantilla generada en: {output_csv}")
    print(f"Audios encontrados: {n_total} | ya etiquetados: {n_labeled} | pendientes: {n_total - n_labeled}")

    if sample_rate_warnings:
        print(
            f"\nADVERTENCIA: {len(sample_rate_warnings)} archivo(s) no tienen el "
            f"sample rate esperado ({SAMPLE_RATE} Hz). Deben reconvertirse antes "
            "de usarse en el pipeline, o el preprocesamiento fallará:"
        )
        for name, actual_sr in sample_rate_warnings:
            print(f"  - {name}: {actual_sr} Hz")

    return output_csv


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Genera la plantilla de metadata.csv a partir de los audios en data/raw/."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Ignora el metadata.csv existente y lo regenera desde cero (pierde etiquetas ya ingresadas).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    try:
        generate_metadata_template(overwrite=args.overwrite)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
