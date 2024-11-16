import torch
from torch import nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.randn(dim,))
        self.dim = dim
    def forward(self, x:torch.Tensor):
        if x.dim() == 1:
            return F.normalize(x, dim = 1) * self.g * (x.shape[1] ** 0.5)
        return F.normalize(x, dim = 1) * self.g.reshape([1, self.dim] + [1] * (x.dim() - 2)) * (x.shape[1] ** 0.5)
    

class AdaLN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.modulator = nn.Linear(dim, dim * 6)

    def forward(self, x):
        x = self.modulator(x)
        return torch.chunk(x, 6, dim=-1)
    
    @staticmethod
    def modulate(x:torch.Tensor, shift:torch.Tensor, scale:torch.Tensor):
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
