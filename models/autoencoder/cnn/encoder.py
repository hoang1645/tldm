import torch
from torch import nn
from einops import rearrange
from typing import Callable, Tuple
from .cnn import ResnetBlock
    
class LatentSpaceEncoder(nn.Module):
    def __init__(self, in_chan:int=3, out_chan:int=32, 
                 conv_init_chan:int=32,
                 activation: Callable[..., torch.Tensor]=torch.nn.functional.relu,
                 chan_mults:Tuple[int, int, int, int]=(1, 2, 4, 8),
                 downscale:int=8,
                 ):
        super().__init__()
        self.conv_init = nn.Conv2d(in_chan, conv_init_chan, 3, padding='same')
        self.layers = nn.Sequential()
        for i, o in zip([1]+list(chan_mults), chan_mults): self.layers.append(ResnetBlock(i * conv_init_chan, o * conv_init_chan))
        self.downsampler = nn.PixelUnshuffle(downscale)
        self.activation = activation
        self.last_conv = nn.LazyConv2d(out_chan, 3, padding='same')
    def forward(self, x:torch.Tensor):
        assert x.dim() == 4
        x = self.conv_init(x)
        x = self.layers(x)
        x = self.activation(x)
        x = self.downsampler(x)
        x = self.last_conv(x)
        return x
    