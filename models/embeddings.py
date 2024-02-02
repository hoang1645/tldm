import torch
from torch import nn 
import math

class TimestepEmbedding(nn.Module):
    def __init__(self, dim:int):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps:torch.Tensor):
        half_dim = self.dim >> 1
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb
