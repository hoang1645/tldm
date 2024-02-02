from torch import nn
import torch
from typing import List


class Block(nn.Module):
    def __init__(self, in_chan:int, out_chan:int, norm_groups:int = 8):
        super().__init__()
        self.proj = nn.Conv2d(in_channels=in_chan, out_channels=out_chan, kernel_size=(3, 3), padding = 1)
        self.norm = nn.GroupNorm(norm_groups, out_chan)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift:List[torch.Tensor]|None = None)->torch.Tensor:
        x = self.proj(x)
        x = self.norm(x)

        if scale_shift is not None:
            scale, shift = scale_shift
            # print(x.shape, scale.shape, shift.shape)
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x

class ResnetBlock(nn.Module):
    def __init__(self, in_chan:int, out_chan:int, *, time_emb_dim = None, groups = 8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_chan * 2)
        ) if time_emb_dim is not None else None

        self.block1 = Block(in_chan, out_chan, norm_groups = groups)
        self.block2 = Block(out_chan, out_chan, norm_groups = groups)
        self.res_conv = nn.Conv2d(in_chan, out_chan, 1) if in_chan != out_chan else nn.Identity()

    def forward(self, x:torch.Tensor, time_emb:None|torch.Tensor = None):
        scale_shift = None
        if self.mlp is not None and time_emb is not None:
            time_emb = self.mlp(time_emb)
            time_emb = time_emb.unsqueeze(2).unsqueeze(3)
            scale_shift = time_emb.chunk(2, dim = 1)

        h = self.block1(x, scale_shift = scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)
    
