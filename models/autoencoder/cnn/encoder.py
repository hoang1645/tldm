import torch
from torch import nn 
import torch.nn.functional as F
from .residual import ResBlock


class LatentSpaceEncoder(nn.Module):
    def __init__(self, n_channels_init:int, in_chan:int=3, out_chan:int=4, num_layers_main:int=3):
        super().__init__()
        self.conv_init = nn.Sequential(
            nn.Conv2d(in_chan, n_channels_init, 7, padding='same'),
            nn.SiLU()
        )

        self.main_sequence_blocks = nn.Sequential(*[
            self.__make_block(n_channels_init << i) for i in range(num_layers_main)
        ])

        self.last_conv = nn.Conv2d(n_channels_init << num_layers_main, out_chan, kernel_size=3, padding=1)
    def __make_block(self, in_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels * 2),
            nn.SiLU(),
            nn.Conv2d(in_channels * 2, in_channels * 2, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(in_channels * 2),
            nn.SiLU()
        )
    
    def forward(self, x):
        x = self.conv_init(x)
        x = self.main_sequence_blocks(x)
        x = self.last_conv(x)

        return x
