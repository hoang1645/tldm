import torch
from torch import nn 
import torch.nn.functional as F



class LatentSpaceEncoder(nn.Module):
    def __init__(self, n_channels_init:int, in_chan:int=3, out_chan:int=32, leaky_relu_slope:float=.2):
        self.leaky_relu_slope = leaky_relu_slope
        self.conv_init = nn.Sequential(
            nn.Conv2d(in_chan, n_channels_init, 7, padding='same'),
            nn.LeakyReLU(self.leaky_relu_slope)
        )
        self.init_block = nn.Sequential(
            nn.Conv2d(n_channels_init, n_channels_init, stride=2, kernel_size=3, padding=1),
            nn.BatchNorm2d(n_channels_init),
            nn.LeakyReLU(self.leaky_relu_slope)
        )

        self.main_sequence_blocks = nn.Sequential(*[
            self.__make_block(n_channels_init << (i+1)) for i in range(3)
        ])

        self.last_conv = nn.Conv2d(n_channels_init << 3, out_chan, kernel_size=3, padding=1)
    def __make_block(self, in_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels * 2),
            nn.LeakyReLU(self.leaky_relu_slope),
            nn.Conv2d(in_channels * 2, in_channels * 2, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(in_channels * 2),
            nn.LeakyReLU(self.leaky_relu_slope)
        )
    
    def forward(self, x):
        x = self.conv_init(x)
        x = self.init_block(x)
        x = self.main_sequence_blocks(x)
        x = self.last_conv(x)

        return x
