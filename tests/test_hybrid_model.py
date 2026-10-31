"""
Pruebas del modelo híbrido CNN-Transformer: verifican que el forward pass
completo (CNN -> proyección -> Transformer -> regresión) funcione end-to-end
con tensores sintéticos, sin depender de audio real.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.cnn_extractor import CNNExtractor  # noqa: E402
from src.models.transformer_block import TransformerBlock, PositionalEncoding  # noqa: E402
from src.models.hybrid_model import HybridCNNTransformer  # noqa: E402


def test_cnn_extractor_output_channels_match_config():
    cnn = CNNExtractor(channels=[1, 8, 16])
    x = torch.randn(2, 1, 64, 64)
    out = cnn(x)

    assert out.shape[0] == 2
    assert out.shape[1] == 16


def test_cnn_extractor_rejects_invalid_channel_config():
    with pytest.raises(ValueError):
        CNNExtractor(channels=[1])


def test_positional_encoding_preserves_shape():
    pe = PositionalEncoding(d_model=32, max_len=100)
    x = torch.randn(4, 20, 32)
    out = pe(x)

    assert out.shape == x.shape


def test_transformer_block_preserves_shape():
    block = TransformerBlock(d_model=32, n_heads=4, n_layers=1, dim_feedforward=64)
    x = torch.randn(4, 10, 32)
    out = block(x)

    assert out.shape == x.shape


def test_transformer_block_rejects_incompatible_heads():
    with pytest.raises(ValueError):
        TransformerBlock(d_model=30, n_heads=4)


def test_hybrid_model_end_to_end_forward():
    model = HybridCNNTransformer(d_model=32)
    model.transformer = TransformerBlock(d_model=32, n_heads=4, n_layers=1, dim_feedforward=64)

    batch_size = 3
    spectrogram = torch.randn(batch_size, 1, 128, 87)  # espectrograma sintético

    output = model(spectrogram)

    assert output.shape == (batch_size,)
    assert torch.isfinite(output).all()


def test_hybrid_model_rejects_wrong_input_dims():
    model = HybridCNNTransformer(d_model=32)
    model.transformer = TransformerBlock(d_model=32, n_heads=4, n_layers=1, dim_feedforward=64)

    invalid_input = torch.randn(3, 128, 87)  # falta la dimensión de canal

    with pytest.raises(ValueError):
        model(invalid_input)
