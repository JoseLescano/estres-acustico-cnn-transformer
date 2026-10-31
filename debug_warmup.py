import torch
from torch.utils.data import DataLoader, Dataset
from src.models.hybrid_model import HybridCNNTransformer
from src.models.transformer_block import TransformerBlock
from src.training.train import _warmup_forward_pass

class D(Dataset):
    def __len__(self):
        return 8
    def __getitem__(self, index):
        return torch.randn(1, 64, 40), torch.rand(1).squeeze()

model = HybridCNNTransformer(d_model=16)
model.transformer = TransformerBlock(d_model=16, n_heads=2, n_layers=1, dim_feedforward=32)
loader = DataLoader(D(), batch_size=4)
print('before', sum(p.numel() for p in model.parameters()))
print('lazy weight before', hasattr(model.input_projection, 'weight'))
print('lazy in_features', getattr(model.input_projection, 'in_features', None))
_warmup_forward_pass(model, loader)
print('after', sum(p.numel() for p in model.parameters()))
print('lazy weight after', hasattr(model.input_projection, 'weight'))
print('lazy in_features after', getattr(model.input_projection, 'in_features', None))
print('weight shape', tuple(getattr(model.input_projection.weight, 'shape', ())))
