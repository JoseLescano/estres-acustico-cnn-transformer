"""
Pruebas unitarias del módulo de generación de espectrogramas de Gabor.
Usan señales sintéticas (senoidales, ruido, silencio) para no depender
de archivos de audio reales en el repositorio.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.gabor_spectrogram import (  # noqa: E402
    compute_gabor_spectrogram,
    normalize_spectrogram,
    pad_or_trim_spectrogram,
)
from src.config import SAMPLE_RATE, N_FFT, HOP_LENGTH  # noqa: E402


def _make_sine_wave(freq_hz: float = 440.0, duration_seconds: float = 1.0) -> np.ndarray:
    t = np.linspace(0, duration_seconds, int(SAMPLE_RATE * duration_seconds), endpoint=False)
    return np.sin(2 * np.pi * freq_hz * t).astype(np.float32)


def test_compute_gabor_spectrogram_output_shape():
    audio = _make_sine_wave(duration_seconds=1.0)
    spectrogram = compute_gabor_spectrogram(audio)

    expected_freq_bins = N_FFT // 2 + 1
    assert spectrogram.shape[0] == expected_freq_bins
    assert spectrogram.shape[1] > 0


def test_compute_gabor_spectrogram_rejects_empty_audio():
    with pytest.raises(ValueError):
        compute_gabor_spectrogram(np.array([], dtype=np.float32))


def test_compute_gabor_spectrogram_rejects_non_1d_audio():
    stereo_audio = np.zeros((SAMPLE_RATE, 2), dtype=np.float32)
    with pytest.raises(ValueError):
        compute_gabor_spectrogram(stereo_audio)


def test_normalize_spectrogram_has_zero_mean_unit_std():
    audio = _make_sine_wave(duration_seconds=1.0)
    spectrogram = compute_gabor_spectrogram(audio)
    normalized = normalize_spectrogram(spectrogram)

    assert abs(normalized.mean()) < 1e-4
    assert abs(normalized.std() - 1.0) < 1e-4


def test_normalize_spectrogram_handles_constant_input():
    constant_spectrogram = np.ones((10, 10), dtype=np.float32) * 5.0
    normalized = normalize_spectrogram(constant_spectrogram)

    assert np.allclose(normalized, 0.0)


def test_pad_or_trim_spectrogram_pads_when_shorter():
    spectrogram = np.ones((16, 5), dtype=np.float32)
    result = pad_or_trim_spectrogram(spectrogram, target_frames=10)

    assert result.shape == (16, 10)
    assert np.all(result[:, 5:] == 0.0)


def test_pad_or_trim_spectrogram_trims_when_longer():
    spectrogram = np.ones((16, 20), dtype=np.float32)
    result = pad_or_trim_spectrogram(spectrogram, target_frames=10)

    assert result.shape == (16, 10)


def test_pad_or_trim_spectrogram_noop_when_equal():
    spectrogram = np.ones((16, 10), dtype=np.float32)
    result = pad_or_trim_spectrogram(spectrogram, target_frames=10)

    assert np.array_equal(result, spectrogram)
