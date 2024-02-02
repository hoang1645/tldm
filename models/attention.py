import torch
from torch import nn
from torch.backends.cuda import sdp_kernel
from .normalization import RootMeanSquaredNorm as RMSNorm
from einops import rearrange

class FlashAttention(nn.Module):
    def __init__(self, dropout: float = 0, flash=False):
        
        super().__init__()
        
        self.dropout = dropout
        if torch.cuda.is_available():
            device_properties = torch.cuda.get_device_properties(torch.device('cuda'))
        
            if device_properties.major >= 8 and flash:
                self.gpu_config = {'enable_flash':True, 'enable_math':False, 'enable_mem_efficient':False}
            else:
                self.gpu_config = {'enable_flash':False, 'enable_math':True, 'enable_mem_efficient':True}
        else: self.gpu_config = {}

        self.non_gpu_config = {'enable_flash':True, 'enable_math':True, 'enable_mem_efficient':True}
    
    def forward(self, q:torch.Tensor, k:torch.Tensor, v:torch.Tensor) -> torch.Tensor:
        _, heads, q_len, _, k_len, is_cuda, device = *q.shape, k.shape[-2], q.is_cuda, q.device

        q, k, v = map(lambda t: t.contiguous(), (q, k, v))

        # Check if there is a compatible device for flash attention

        config = self.gpu_config if is_cuda else self.non_gpu_config

        
        with sdp_kernel(**config):
            out = torch.nn.functional.scaled_dot_product_attention(
                q, k, v,
                dropout_p = self.dropout if self.training else 0.
            )

        return out
    

class LinearAttention(nn.Module):
    def __init__(
        self,
        dim,
        heads = 4,
        dim_head = 32
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.norm = RMSNorm(dim)
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            RMSNorm(dim)
        )

    def forward(self, x):
        b, c, h, w = x.shape

        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)

        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        q = q * self.scale

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h = self.heads, x = h, y = w)
        return self.to_out(out)
    
class FlashMultiHeadAttn(nn.Module):
    def __init__(
        self,
        dim:int,
        heads = 4,
        dim_head = 32,
    ):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads

        self.norm = RMSNorm(dim)
        self.attend = FlashAttention(flash=True)

        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape

        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h (x y) c', h = self.heads), qkv)

        out = self.attend(q, k, v)

        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = h, y = w)
        out = self.to_out(out)
        return out
