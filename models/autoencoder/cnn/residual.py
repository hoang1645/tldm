import torch
from torch import nn
import torch.nn.functional as F
from typing import Callable

class ResBlock(nn.Module):
    def __init__(self, n_channels, n_residual_blocks, activation:Callable[..., torch.Tensor]=F.relu):
        super().__init__()
        self.activation = activation
        self.blocks = nn.ModuleList([self.__make_block(n_channels) for _ in range(n_residual_blocks)])
    def __make_block(self, n_channels):
        return nn.Sequential(nn.Conv2d(n_channels, n_channels, 3, padding=1),
                             nn.BatchNorm2d(n_channels),
                             nn.SiLU())
    def forward(self, x: torch.Tensor)->torch.Tensor:
        x0 = x
        for block in self.blocks:
            x = x + self.activation(block(x)) 
        return x0 + x