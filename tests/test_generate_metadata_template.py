"""
Pruebas del script de generación de metadata.csv. Usa una carpeta temporal
con archivos .wav sintéticos reales (generados con soundfile), sin
depender de audio de campo.
"""

import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.generate_metadata_template import (  # noqa: E402
    find_audio_files,
    validate_sample_rate,
    generate_metadata_template,
)
from src.config import SAMPLE_RATE  # noqa: E402


def _write_dummy_wav(path: Path, sample_rate: int, duration_seconds: float = 1.0) -> None:
    n_samples = int(sample_rate * duration_seconds)
    audio = np.zeros(n_samples, dtype=np.float32)
    sf.write(str(path), audio, sample_rate)


def test_find_audio_files_returns_only_wav(tmp_path):
    _write_dummy_wav(tmp_path / "audio1.wav", SAMPLE_RATE)
    _write_dummy_wav(tmp_path / "audio2.wav", SAMPLE_RATE)
    (tmp_path / "notes.txt").write_text("no es audio")

    found = find_audio_files(tmp_path)

    assert len(found) == 2
    assert all(p.suffix == ".wav" for p in found)


def test_find_audio_files_raises_if_dir_missing(tmp_path):
    missing_dir = tmp_path / "no_existe"
    try:
        find_audio_files(missing_dir)
        assert False, "Debió lanzar FileNotFoundError"
    except FileNotFoundError:
        pass


def test_validate_sample_rate_detects_mismatch(tmp_path):
    wrong_sr_file = tmp_path / "audio_44100.wav"
    _write_dummy_wav(wrong_sr_file, sample_rate=44100)

    is_valid, actual_sr = validate_sample_rate(wrong_sr_file, expected_sr=SAMPLE_RATE)

    assert is_valid is False
    assert actual_sr == 44100


def test_validate_sample_rate_accepts_match(tmp_path):
    correct_file = tmp_path / "audio_correct.wav"
    _write_dummy_wav(correct_file, sample_rate=SAMPLE_RATE)

    is_valid, actual_sr = validate_sample_rate(correct_file, expected_sr=SAMPLE_RATE)

    assert is_valid is True
    assert actual_sr == SAMPLE_RATE


def test_generate_metadata_template_creates_csv_with_empty_labels(tmp_path):
    _write_dummy_wav(tmp_path / "a.wav", SAMPLE_RATE)
    _write_dummy_wav(tmp_path / "b.wav", SAMPLE_RATE)

    output_csv = tmp_path / "metadata.csv"
    generate_metadata_template(raw_dir=tmp_path, output_csv=output_csv)

    content = output_csv.read_text(encoding="utf-8")
    assert "filename,stress_index" in content
    assert "a.wav" in content
    assert "b.wav" in content


def test_generate_metadata_template_preserves_existing_labels(tmp_path):
    _write_dummy_wav(tmp_path / "a.wav", SAMPLE_RATE)
    _write_dummy_wav(tmp_path / "b.wav", SAMPLE_RATE)

    output_csv = tmp_path / "metadata.csv"

    # Primera corrida: sin etiquetas
    generate_metadata_template(raw_dir=tmp_path, output_csv=output_csv)

    # El investigador etiqueta manualmente "a.wav"
    content = output_csv.read_text(encoding="utf-8")
    content = content.replace("a.wav,", "a.wav,0.42")
    output_csv.write_text(content, encoding="utf-8")

    # Se agrega un tercer audio nuevo y se vuelve a correr el script
    _write_dummy_wav(tmp_path / "c.wav", SAMPLE_RATE)
    generate_metadata_template(raw_dir=tmp_path, output_csv=output_csv)

    final_content = output_csv.read_text(encoding="utf-8")
    assert "a.wav,0.42" in final_content  # la etiqueta previa se conservó
    assert "c.wav" in final_content        # el nuevo audio se agregó


def test_generate_metadata_template_overwrite_resets_labels(tmp_path):
    _write_dummy_wav(tmp_path / "a.wav", SAMPLE_RATE)
    output_csv = tmp_path / "metadata.csv"

    generate_metadata_template(raw_dir=tmp_path, output_csv=output_csv)
    content = output_csv.read_text(encoding="utf-8").replace("a.wav,", "a.wav,0.9")
    output_csv.write_text(content, encoding="utf-8")

    generate_metadata_template(raw_dir=tmp_path, output_csv=output_csv, overwrite=True)

    final_content = output_csv.read_text(encoding="utf-8")
    assert "a.wav,0.9" not in final_content
