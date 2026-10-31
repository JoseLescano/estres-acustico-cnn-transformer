"""
Pruebas básicas de sanidad sobre la configuración del proyecto.
Verifican que las rutas y parámetros definidos en src/config.py sean
consistentes antes de construir el pipeline de datos y el modelo.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402


def test_root_dir_exists():
    assert config.ROOT_DIR.exists()


def test_data_subdirectories_are_defined_under_data_dir():
    for subdir in [
        config.RAW_DATA_DIR,
        config.INTERIM_DATA_DIR,
        config.PROCESSED_DATA_DIR,
        config.EXTERNAL_DATA_DIR,
    ]:
        assert subdir.parent == config.DATA_DIR


def test_dataset_split_sums_to_one():
    total = config.TRAIN_SPLIT + config.VAL_SPLIT + config.TEST_SPLIT
    assert abs(total - 1.0) < 1e-9


def test_transformer_d_model_divisible_by_heads():
    assert config.TRANSFORMER_D_MODEL % config.TRANSFORMER_N_HEADS == 0


def test_device_is_valid():
    assert config.DEVICE in ("cuda", "cpu")
