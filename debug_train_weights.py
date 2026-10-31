import torch
from torch.utils.data import DataLoader, Dataset
from src.models.hybrid_model import HybridCNNTransformer
from src.models.transformer_block import TransformerBlock
from src.training.train import _warmup_forward_pass, _run_epoch

class D(Dataset):
    def __len__(self):
        return 8
    def __getitem__(self, index):
        spectrogram = torch.randn(1, 64, 40)
        label = torch.rand(1).squeeze()
        return spectrogram, label

model = HybridCNNTransformer(d_model=16)
model.transformer = TransformerBlock(d_model=16, n_heads=2, n_layers=1, dim_feedforward=32)
loader = DataLoader(D(), batch_size=4)
_warmup_forward_pass(model, loader)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = torch.nn.MSELoss()
weights_before = model.regression_head[0].weight.clone()
loss, metrics = _run_epoch(model, loader, criterion, optimizer)
weights_after = model.regression_head[0].weight
print('loss', loss)
print('metrics', metrics)
print('allclose', torch.allclose(weights_before, weights_after))
print('max_abs_delta', (weights_after - weights_before).abs().max().item())
print('before', weights_before.flatten()[:5])
print('after', weights_after.flatten()[:5])
