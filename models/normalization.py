import torch
from torch import nn 
import torch.nn.functional as F

class RootMeanSquaredNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.randn(dim,))
        self.dim = dim
    def forward(self, x:torch.Tensor):
        if x.dim() == 1:
            return F.normalize(x, dim = 1) * self.g * (x.shape[1] ** 0.5)
        return F.normalize(x, dim = 1) * self.g.reshape([1, self.dim] + [1] * (x.dim() - 2)) * (x.shape[1] ** 0.5)
    