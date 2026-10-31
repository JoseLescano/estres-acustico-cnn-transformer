"""
Generación de espectrogramas de Gabor a partir de señales de audio.

El espectrograma de Gabor usa una ventana gaussiana en lugar de la ventana
rectangular/Hann típica de un STFT convencional, lo que produce una mejor
resolución conjunta tiempo-frecuencia (menor incertidumbre de Heisenberg
tiempo-frecuencia) — de ahí su uso para caracterizar ruido no estacionario,
donde los eventos transitorios (bocinas, impactos, maquinaria) tienen
bordes temporales que un STFT con ventana Hann tiende a difuminar.
"""

from __future__ import annotations

import numpy as np
from scipy.signal.windows import gaussian
from scipy.signal import stft

from src.config import (
    SAMPLE_RATE,
    N_FFT,
    HOP_LENGTH,
    GABOR_SIGMA,
)


def _gabor_window(win_length: int, sigma: float) -> np.ndarray:
    """
    Construye una ventana gaussiana normalizada (base del espectrograma de Gabor).

    Args:
        win_length: longitud de la ventana en muestras.
        sigma: desviación estándar de la gaussiana, en muestras.

    Returns:
        Array 1D de longitud `win_length` con la ventana gaussiana.
    """
    window = gaussian(win_length, std=sigma, sym=True)
    # Normalización de energía para que distintas ventanas sean comparables
    norm = np.sqrt(np.sum(window ** 2))
    if norm == 0:
        raise ValueError("La ventana gaussiana generada tiene energía nula.")
    return window / norm


def compute_gabor_spectrogram(
    audio: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
    sigma: float = GABOR_SIGMA,
    log_scale: bool = True,
) -> np.ndarray:
    """
    Calcula el espectrograma de Gabor de una señal de audio 1D.

    Args:
        audio: señal de audio mono, shape (n_samples,).
        sample_rate: frecuencia de muestreo en Hz.
        n_fft: tamaño de la ventana FFT (y de la ventana gaussiana).
        hop_length: desplazamiento entre ventanas consecutivas.
        sigma: desviación estándar de la ventana gaussiana.
        log_scale: si es True, aplica escala logarítmica (dB) a la magnitud.

    Returns:
        Array 2D (n_freq_bins, n_time_frames) con la magnitud del
        espectrograma de Gabor.

    Raises:
        ValueError: si `audio` está vacío o no es un array 1D.
    """
    if audio.ndim != 1:
        raise ValueError(f"Se esperaba audio mono (1D), se recibió shape {audio.shape}")
    if audio.size == 0:
        raise ValueError("La señal de audio está vacía.")

    window = _gabor_window(n_fft, sigma)
    noverlap = n_fft - hop_length

    _freqs, _times, stft_matrix = stft(
        audio,
        fs=sample_rate,
        window=window,
        nperseg=n_fft,
        noverlap=noverlap,
        boundary=None,
        padded=False,
    )

    magnitude = np.abs(stft_matrix)

    if log_scale:
        # Piso numérico para evitar log(0)
        eps = 1e-10
        magnitude = 20.0 * np.log10(magnitude + eps)

    return magnitude.astype(np.float32)


def normalize_spectrogram(spectrogram: np.ndarray) -> np.ndarray:
    """
    Normaliza un espectrograma a media 0 y desviación estándar 1 (z-score),
    usado antes de alimentar el espectrograma al bloque CNN.

    Args:
        spectrogram: array 2D del espectrograma.

    Returns:
        Espectrograma normalizado, mismo shape que la entrada.
    """
    mean = spectrogram.mean()
    std = spectrogram.std()
    if std < 1e-8:
        # Espectrograma prácticamente constante (p. ej. silencio total)
        return spectrogram - mean
    return (spectrogram - mean) / std


def pad_or_trim_spectrogram(spectrogram: np.ndarray, target_frames: int) -> np.ndarray:
    """
    Ajusta un espectrograma a un número fijo de frames temporales,
    recortando o rellenando con ceros por el lado derecho.

    Necesario porque los segmentos de audio pueden variar levemente en
    longitud y el modelo requiere tensores de tamaño fijo por batch.

    Args:
        spectrogram: array 2D (n_freq_bins, n_time_frames).
        target_frames: número deseado de frames temporales.

    Returns:
        Array 2D (n_freq_bins, target_frames).
    """
    n_freq_bins, n_frames = spectrogram.shape

    if n_frames == target_frames:
        return spectrogram

    if n_frames > target_frames:
        return spectrogram[:, :target_frames]

    pad_width = target_frames - n_frames
    return np.pad(spectrogram, ((0, 0), (0, pad_width)), mode="constant", constant_values=0.0)
