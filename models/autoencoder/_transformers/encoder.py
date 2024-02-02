import torch
from torch import nn
from einops import rearrange
from typing import Callable, Tuple
from .attention import TransformerMultiHeadAttention


class PatchEmbedding(nn.Module):
    def __init__(self, kernel_size:int, in_channels:int=3, out_channels:int=32):
        super().__init__()
        self.embedding = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                                   kernel_size=kernel_size, stride=kernel_size)

    def forward(self, x:torch.Tensor):
        return self.embedding(x).flatten(2)
    


    
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model:int, n_heads:int, d_ff:int, dropout:float, activation:Callable[..., torch.Tensor],
                 norm_first:bool=False):
        super().__init__()
        self.ffn1 = nn.Conv1d(d_model, d_ff, kernel_size=1)
        self.ffn2 = nn.Conv1d(d_ff, d_model, kernel_size=1)
        self.norm = nn.LayerNorm(d_model)
        self.attn = TransformerMultiHeadAttention(d_model, n_heads, d_head = d_model//n_heads, dropout=dropout, flash=True)
        self.activation = activation
        self.dropout = dropout
        self.norm_first = norm_first

    def forward(self, x:torch.Tensor): 
        if self.norm_first:
            x = x.transpose(-1, -2)
            x = self.norm(x)
            x = x.transpose(-1, -2)
            
        
        x = self.attn(x)
        x = torch.nn.functional.dropout(x, self.dropout, training=self.training)
        x = self.ffn1(x)
        x = self.activation(x)
        x = torch.nn.functional.dropout(x, self.dropout, training=self.training)
        x = self.ffn2(x)
        x = self.activation(x)

        if not self.norm_first:
            x = x.transpose(-1, -2)
            x = self.norm(x)
            x = x.transpose(-1, -2)
            
        
        return x
    
class LatentSpaceEncoder(nn.Module):
    def __init__(self, kernel_size:int, n_attn_heads:int, d_model:int, n_layers:int, dropout:float=0,
                 activation:Callable[..., torch.Tensor]=torch.nn.functional.relu):
        super().__init__()
        self.embedding = PatchEmbedding(kernel_size=kernel_size, out_channels=d_model)
        self.kernel_size = kernel_size
        self.transformer_layers = nn.Sequential()
        for _ in range(n_layers): self.transformer_layers.append(TransformerEncoderLayer(d_model, n_attn_heads, d_model<<2,
                                                                                         dropout, activation))
        self.pooler = nn.Conv1d(d_model, d_model, 1)
        self.downsampler = nn.Conv1d(d_model, d_model // kernel_size, 1)

    def forward(self, x:torch.Tensor):
        assert x.dim() == 4
        h = x.shape[2]
        patch_size_h = h // self.kernel_size
        
        x = self.embedding(x)
        x = self.transformer_layers(x)
        x = self.pooler(x)
        x = self.downsampler(x)

        x = rearrange(x, "b c (h w) -> b c h w", h=patch_size_h)
        return x
    