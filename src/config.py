"""
Configuración centralizada del proyecto.

Todas las rutas, hiperparámetros y constantes usadas en el pipeline
(preprocesamiento, entrenamiento, evaluación) se definen aquí para
evitar valores mágicos dispersos en el código.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Rutas del proyecto
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_SAVED_DIR = ROOT_DIR / "models_saved"
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# ---------------------------------------------------------------------------
# Parámetros de audio
# ---------------------------------------------------------------------------
SAMPLE_RATE = 22050          # Hz — frecuencia de muestreo estándar para análisis acústico
AUDIO_DURATION_SECONDS = 4.0  # duración fija de cada segmento de audio analizado
N_FFT = 1024                  # tamaño de la ventana FFT para el espectrograma de Gabor
HOP_LENGTH = 256               # desplazamiento entre ventanas consecutivas
N_MELS = 128                    # bandas de frecuencia (si se usa escala Mel como referencia)
GABOR_SIGMA = 6.0                 # desviación estándar de la ventana gaussiana (espectrograma de Gabor)
SOUND_LEVEL_REFERENCE_DB = 94.0   # calibración provisional: amplitud digital 1.0 equivale a 94 dB SPL

# ---------------------------------------------------------------------------
# Hiperparámetros del modelo híbrido CNN-Transformer
# ---------------------------------------------------------------------------
CNN_CHANNELS = [1, 16, 32, 64]      # canales de entrada/salida de cada bloque convolucional
CNN_KERNEL_SIZE = 3
CNN_POOL_SIZE = 2

TRANSFORMER_D_MODEL = 128            # dimensión de los embeddings que entran al Transformer
TRANSFORMER_N_HEADS = 4               # número de cabezas de atención
TRANSFORMER_N_LAYERS = 2               # capas del encoder Transformer
TRANSFORMER_DIM_FEEDFORWARD = 256
TRANSFORMER_DROPOUT = 0.1

# ---------------------------------------------------------------------------
# Hiperparámetros de entrenamiento
# ---------------------------------------------------------------------------
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
NUM_EPOCHS = 50
WEIGHT_DECAY = 1e-5
RANDOM_SEED = 42

# División del dataset
TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

# ---------------------------------------------------------------------------
# Dispositivo de cómputo
# ---------------------------------------------------------------------------
try:
    import torch  # noqa: E402

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except (ImportError, OSError):
    # Permite importar config.py (y módulos que solo necesitan rutas/constantes,
    # como el generador de metadata.csv) en entornos sin torch instalado o con
    # una instalación de torch incompleta/rota. Cualquier código que sí requiera
    # entrenar/inferir con el modelo seguirá fallando explícitamente al intentar
    # usar torch directamente.
    DEVICE = "cpu"
