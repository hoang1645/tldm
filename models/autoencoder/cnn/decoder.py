import torch
from torch import nn 
import torch.nn.functional as F
from .residual import ResBlock


class LatentSpaceDecoder(nn.Module):
    def __init__(self, n_channels_init:int, in_chan:int=32, out_chan:int=3):
        super().__init__()
        self.init_conv = nn.Sequential(
            nn.Conv2d(in_chan, n_channels_init, 3, padding=1),
            nn.PReLU()
        )
        self.residual_blocks = nn.Sequential(*[
            self.__make_block(n_channels_init << i, 2*i + 2) for i in range(3)
        ])
        self.final_block = nn.Sequential(
            ResBlock(n_channels_init << 3, 8, F.relu6),
            self.__upscale_block(n_channels_init << 3, n_channels_init << 3)
        )
        self.final_conv = nn.Sequential(
            nn.Conv2d(n_channels_init << 3, out_chan, 3, padding='same'),
        )


    def __make_block(self, in_channels, n_residual_blocks):
        return nn.Sequential(
            ResBlock(in_channels, n_residual_blocks, F.relu6),
            self.__upscale_block(in_channels, in_channels << 1)
        )
    
    def __upscale_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels << 2, 3, padding=1),
            nn.PixelShuffle(2),
            nn.PReLU()
        )
    
    def forward(self, x):
        x = self.init_conv(x)
        x = self.residual_blocks(x)
        x = self.final_block(x)
        x = self.final_conv(x)
        return x