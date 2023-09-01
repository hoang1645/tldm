import torch
from torch import nn
from einops import rearrange
from typing import Callable, Tuple
from .cnn import ResnetBlock
    
class LatentSpaceDecoder(nn.Module):
    def __init__(self, in_chan:int=32, out_chan:int=3, 
                 conv_init_chan:int=32,
                 activation: Callable[..., torch.Tensor]=torch.nn.functional.relu,
                 chan_mults:Tuple[int, int, int, int]=(8, 4, 2, 1),
                 upscale:int=8,
                 ):
        super().__init__()
        self.conv_init = nn.Conv2d(in_chan, conv_init_chan * chan_mults[0] * upscale ** 2, 3, padding='same')
        self.layers = nn.Sequential()
        for i, o in zip (chan_mults, list(chan_mults)[1:] + [1]): self.layers.append(ResnetBlock(i * conv_init_chan, o * conv_init_chan))
        self.upsampler = nn.PixelShuffle(upscale)
        self.activation = activation
        self.last_conv = nn.LazyConv2d(out_chan, 3, padding='same')
    def forward(self, x:torch.Tensor):
        assert x.dim() == 4
        x = self.conv_init(x)
        x = self.upsampler(x)
        x = self.layers(x)
        x = self.activation(x)
        
        x = self.last_conv(x)
        return x
    