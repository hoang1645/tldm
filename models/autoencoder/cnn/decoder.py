import torch
from torch import nn 
import torch.nn.functional as F



class LatentSpaceDecoder(nn.Module):
    def __init__(self, n_channels_init:int, in_chan:int=32, out_chan:int=3, n_blocks:int=8):
        super().__init__()
        self.init_conv = nn.Sequential(
            nn.Conv2d(in_chan, n_channels_init, 3, padding=1),
            nn.PReLU()
        )
        self.residual_blocks = nn.ModuleList([
            self.__make_block(n_channels_init) for _ in range(n_blocks)
        ])
        self.out_residual = nn.Sequential(
            nn.Conv2d(n_channels_init, n_channels_init, 3, padding=1),
            nn.BatchNorm2d(n_channels_init)
        )
        self.upscaler = nn.Sequential(*[self.__upscale_block(n_channels_init) for _ in range(4)])
        self.final_conv = nn.Conv2d(n_channels_init, 3, 9, padding='same')


    def __make_block(self, in_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.PReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels)
        )
    def __upscale_block(self, in_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels << 2, 3, padding=1),
            nn.PixelShuffle(2),
            nn.PReLU()
        )
    
    def forward(self, x):
        x = self.init_conv(x)
        x_1 = x
        for block in self.residual_blocks:
            x = x + block(x)
        x = x_1 + self.out_residual(x)
        x = self.upscaler(x)
        x = self.final_conv(x)
        return x