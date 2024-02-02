import torch
from torch import nn
from einops import rearrange
from typing import Callable, Tuple
from .attention import TransformerMultiHeadAttention


class Upscaler(nn.Module):
    def __init__(self, upscale_factor:int, in_channels:int=32, out_channels:int=3):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels * upscale_factor**2, 3, padding='same')
        self.upscale = nn.PixelShuffle(upscale_factor)

    def forward(self, x:torch.Tensor):
        return self.upscale(self.conv(x))


    
class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model:int, n_heads:int, d_ff:int, dropout:float, activation:Callable[..., torch.Tensor],
                 norm_first:bool=False):
        super().__init__()
        self.ffn1 = nn.Conv1d(d_model, d_ff, kernel_size=1)
        self.ffn2 = nn.Conv1d(d_ff, d_model, kernel_size=1)
        self.norm = nn.LayerNorm(d_model)
        self.attn = TransformerMultiHeadAttention(d_model, n_heads, 
                                                  d_head = d_model//n_heads, dropout=dropout, flash=True)
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
    
class LatentSpaceDecoder(nn.Module):
    def __init__(self, upscale:int, n_attn_heads:int, d_model:int, n_layers:int, dropout:float=0,
                 activation:Callable[..., torch.Tensor]=torch.nn.functional.relu):
        super().__init__()
        self.upscaler = Upscaler(upscale_factor=upscale, in_channels=d_model)
        self.transformer_layers = nn.Sequential()
        self.undownsampler = nn.Conv1d(d_model // upscale, d_model, 1)
        for _ in range(n_layers): self.transformer_layers.append(TransformerDecoderLayer(d_model, n_attn_heads, d_model<<2,
                                                                                         dropout, activation))

    def forward(self, x:torch.Tensor):
        assert x.dim() == 4
        h = x.shape[2]
        x = rearrange(x, "b c h w -> b c (h w)")
        x = self.undownsampler(x)
        x = self.transformer_layers(x)

        x = rearrange(x, "b c (h w) -> b c h w", h=h)
        x = self.upscaler(x)
        return x
    