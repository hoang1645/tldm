import torch
from torch import nn
import torch.nn.functional as F
from xformers.components import MultiHeadDispatch
from xformers.components.attention import build_attention



class MemoryEfficientAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.attention = build_attention({
            "name": "scaled_dot_product",
            "dropout": dropout,
            "causal": False,  # Set to True if using causal attention
            "requires_input_projection": False
        })
        self.attention.requires_input_projection = False
        self.multihead = MultiHeadDispatch(
            dim_model=embed_dim,
            num_heads=num_heads,
            attention=self.attention,
        )

    def forward(self, x, k=None, v=None, attention_mask=None):
        return self.multihead(x, k, v, attention_mask)

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
    def __init__(self, dim, components=6):
        super().__init__()
        self.modulator = nn.Linear(dim, dim * components)
        self.cross_attn = MemoryEfficientAttention(dim, 1, 0.)
        self.components = components
    def forward(self, x, conditioning):
        x = self.cross_attn(x, conditioning, conditioning)
        x = self.modulator(x)
        return torch.chunk(x, self.components, dim=-1)
    
    @staticmethod
    def modulate(x:torch.Tensor, shift:torch.Tensor, scale:torch.Tensor):
        return x * (1 + scale) + shift
