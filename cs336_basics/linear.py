import torch
from einops import rearrange, einsum

class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, weights, device=None, dtype=None):
        self.in_features = in_features
        self.out_features = out_features
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.weight = weights
        # no bias for now


    def forward(self, input: torch.Tensor) -> torch.Tensor:

        return einsum(input, self.weight, "batch seq in_features, out_features in_features -> batch seq out_features") 