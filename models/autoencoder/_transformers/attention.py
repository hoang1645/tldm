import torch
from torch import nn
from torch.backends.cuda import sdp_kernel
from xformers.ops import memory_efficient_attention
from einops import rearrange
from typing import Callable

class TransformerMultiHeadAttention(nn.Module):
    def __init__(self, dim, n_heads:int=4, d_head:int=32, dropout: float = 0, flash=False, attn_mask=None):
        super().__init__()
        self.n_heads = n_heads
        self.dropout = dropout
        self.dim = dim
        self.attn_mask = attn_mask
        if torch.cuda.is_available():
            device_properties = torch.cuda.get_device_properties(torch.device('cuda'))
        
            if device_properties.major >= 8 and flash:
                self.gpu_config = {'enable_flash':True, 'enable_math':False, 'enable_mem_efficient':False}
            else:
                self.gpu_config = {'enable_flash':False, 'enable_math':True, 'enable_mem_efficient':True}
        else: self.gpu_config = {}

        self.non_gpu_config = {'enable_flash':True, 'enable_math':True, 'enable_mem_efficient':True}
        hidden_dim = d_head * n_heads

        self.norm = nn.LayerNorm(dim)

        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv1d(hidden_dim, dim, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        is_cuda = x.is_cuda
        x = x.transpose(-1, -2)
        x = self.norm(x)
        x = x.transpose(-1, -2)
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) p -> b p h c', h = self.n_heads).contiguous(), qkv)

        # Check if there is a compatible device for flash attention

        config = self.gpu_config if is_cuda else self.non_gpu_config

        
        # with sdp_kernel(**config):
        x = memory_efficient_attention(
            q, k, v,
            p = self.dropout if self.training else 0.,
            attn_bias=self.attn_mask
        )

        x = rearrange(x, 'b p h c -> b (h c) p')
        x = self.to_out(x)
        return x