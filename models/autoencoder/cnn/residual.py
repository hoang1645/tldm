import torch
from torch import nn
import torch.amp
from flash_attn.modules.mha import FlashSelfAttention
from einops import rearrange


class ResBlock(nn.Module):
    def __init__(self, channels_in:int, channels_out:int, num_cnn_layers:int=2, groups:int=32):
        """Multi-purpose residual block"""
        super().__init__()
        self.channels_in = channels_in
        self.channels_out = channels_out
        self.num_cnn_layers = num_cnn_layers
        self.num_groups = groups


        # norm first
        norms = [nn.GroupNorm(self.num_groups, self.channels_in)]
        convs = [nn.Conv2d(self.channels_in, self.channels_out, 3, padding=1)]

        for _ in range(num_cnn_layers - 1):
            norms.append(nn.GroupNorm(self.num_groups, self.channels_out))
            convs.append(nn.Conv2d(self.channels_out, self.channels_out, 3, padding=1))

        self.norms = nn.ModuleList(norms)
        self.convs = nn.ModuleList(convs)
        self.act = nn.SiLU()

        self.conv_l2 = nn.Conv2d(self.channels_in, self.channels_out, 3, padding=1)

    def forward(self, x:torch.Tensor)->torch.Tensor:
        x0 = self.conv_l2(x)
        for i, (norm, conv) in enumerate(zip(self.norms, self.convs)):
            if i == 0: 
                x = conv(norm(x))
            else:
                x += conv(norm(x))
        x = self.act(x) + x0
        return x


class EncoderResBlock(nn.Module):
    def __init__(self, channels_in:int, channels_out:int, num_res_blocks:int=2, num_cnn_layers:int=2, groups:int=32, downsample:bool=True):
        super().__init__()
        self.res_blocks = nn.Sequential(*([ResBlock(channels_in, channels_out, num_cnn_layers, groups)] + 
                                          [ResBlock(channels_out, channels_out, num_cnn_layers, groups) for _ in range(num_res_blocks - 1)]))
        self.downsampler = None
        if downsample:
            self.downsampler = nn.Conv2d(channels_out, channels_out, kernel_size=3, stride=2, padding=1)
    def forward(self, x:torch.Tensor)->torch.Tensor:
        x = self.res_blocks(x)
        if self.downsampler:
            x = self.downsampler(x)
        
        return x

class DecoderResBlock(nn.Module):
    def __init__(self, channels_in:int, channels_out:int, num_res_blocks:int=2, num_cnn_layers:int=2, groups:int=32, upsample:bool=True, attention:bool=True, num_heads:int=4):
        super().__init__()
        self.res_blocks = nn.Sequential(*([ResBlock(channels_in, channels_out, num_cnn_layers, groups)] + 
                                          [ResBlock(channels_out, channels_out, num_cnn_layers, groups) for _ in range(num_res_blocks - 1)]))
        self.upsampler = None
        self.qkv = None
        self.flatten = None
        self.attn = None
        self.num_heads = num_heads
        if upsample:
            self.upsampler = nn.Sequential(nn.Conv2d(channels_out, channels_out * 4, 1), nn.PixelShuffle(2))
        if attention:
            self.attn = FlashSelfAttention()
            self.flatten = nn.Flatten(start_dim=2)
            self.qkv = nn.Linear(channels_out, channels_out * 3)
    
    
    def _attn(self, x:torch.Tensor)->torch.Tensor:
        assert self.attn is not None
        old_shape = x.shape
        x = self.flatten(x).transpose(-1, -2)
        x = self.qkv(x)
        q, k, v = torch.chunk(x, 3, dim=-1) # 3x (B, (H * W), C)
        q = rearrange(q, "B C (H D) -> B C H D", H = self.num_heads)
        k = rearrange(k, "B C (H D) -> B C H D", H = self.num_heads)
        v = rearrange(v, "B C (H D) -> B C H D", H = self.num_heads)
        with torch.cuda.amp.autocast(False):
            x = torch.stack([q, k, v], dim=2).bfloat16()
            x = self.attn(x).float()
        x = x.transpose(-1, -2)
        x = torch.reshape(x, old_shape)
        return x
    
    def forward(self, x:torch.Tensor)->torch.Tensor:
        x = self.res_blocks(x)
        
        if self.attn:
            x = self._attn(x)

        if self.upsampler:
            x = self.upsampler(x)
            
        return x
    


class MiddleBlock(nn.Module):
    def __init__(self, channels:int, num_heads=4):
        super().__init__()
        self.res1 = ResBlock(channels, channels)
        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Linear(channels, channels * 3)
        self.flatten = nn.Flatten(2)
        self.attn = FlashSelfAttention()
        self.out = nn.Linear(channels, channels)
        self.res2 = ResBlock(channels, channels)
        self.num_heads = num_heads
    
    def _attn(self, x:torch.Tensor)->torch.Tensor:
        old_shape = x.shape
        x = self.flatten(x).transpose(-1, -2)
        x = self.qkv(x)
        q, k, v = torch.chunk(x, 3, dim=-1) # 3x (B, (H * W), C)
        q = rearrange(q, "B C (H D) -> B C H D", H = self.num_heads)
        k = rearrange(k, "B C (H D) -> B C H D", H = self.num_heads)
        v = rearrange(v, "B C (H D) -> B C H D", H = self.num_heads)
        with torch.cuda.amp.autocast(False):
            x = torch.stack([q, k, v], dim=2).bfloat16()
            x = self.attn(x).float()
        x = x.transpose(-1, -2)
        x = torch.reshape(x, old_shape)
        return x
    
    
    def forward(self, x:torch.Tensor)->torch.Tensor:
        x = self._attn(x)
        x = rearrange(x, "b c h w -> b h w c")
        x = self.out(x)
        x = rearrange(x, 'b h w c -> b c h w')
        return self.res2(self.norm(self.res1(x)))