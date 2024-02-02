import torch
from torch import nn 
import torch.nn.functional as F
from .cnn.encoder import LatentSpaceEncoder
from .cnn.decoder import LatentSpaceDecoder
from typing import Any, Callable
from vector_quantize_pytorch import VectorQuantize
from abc import abstractmethod
from einops import rearrange
from torch.cuda.amp.autocast_mode import autocast


    
class Autoencoder(nn.Module):
    def __init__(self, n_channels_init:int, latent_space_channel_dim:int=32):
        super().__init__()
        self.encoder = LatentSpaceEncoder(n_channels_init, out_chan=latent_space_channel_dim)
        self.decoder = LatentSpaceDecoder(n_channels_init, in_chan=latent_space_channel_dim)
        
    def encode(self, x:torch.Tensor): return self.encoder(x)
    def decode(self, x:torch.Tensor): return self.decoder(x)
    def forward(self, x:torch.Tensor):
        return self.encoder(self.decoder(x))
    
    @abstractmethod
    def reg_loss(self, _input:torch.Tensor):
        raise NotImplementedError()

class AutoencoderKL(Autoencoder):
    def __init__(self,  n_channels_init:int, latent_space_channel_dim:int=32,
                 kl_penalty:float=1e-6):
        super().__init__(n_channels_init, latent_space_channel_dim)
        self.regularization_loss = nn.KLDivLoss(False)
        self.kl_penalty = kl_penalty
        self.device = 'cuda'
    
    @autocast(enabled=False)
    def reg_loss(self, _input:torch.Tensor):
        target = F.softmax(torch.randn_like(_input), dim=0)
        return self.kl_penalty * self.regularization_loss.forward(F.log_softmax(_input, dim=0), target)
    
class AutoencoderVQ(Autoencoder):
    def __init__(self, n_channels_init:int, latent_space_channel_dim:int=32,
                 codebook_size:int=512):
        super().__init__(n_channels_init, latent_space_channel_dim)
        quant_dim = latent_space_channel_dim
        self.codebook_size = codebook_size
        self.vquantizer = VectorQuantize(quant_dim, self.codebook_size, channel_last=False, ema_update=True)
        self.quant_conv = nn.Conv2d(latent_space_channel_dim, quant_dim, 1)
        self.post_quant_conv = nn.Conv2d(quant_dim, latent_space_channel_dim, 1)
        self.device = 'cuda'

    def encode(self, x:torch.Tensor):
        dtype = x.dtype
        x = self.encoder(x)
        h = x.shape[2]
        x = self.quant_conv(x)
        x = rearrange(x, "b c h w -> b c (h w)")
        x = x.float()
        x, idx, loss = self.vquantizer.forward(x)
        x = x.to(dtype)
        loss = loss.to(dtype)
        x = rearrange(x, "b c (h w) -> b c h w", h=h)
        return x, idx, loss
    
    def decode(self, x:torch.Tensor):
        x = self.post_quant_conv(x)
        x = self.decoder(x)
        return x
    
    def forward(self, x:torch.Tensor):
        x, idx, loss = self.encode(x)
        x = self.decode(x)
        return x, idx, loss
    
    def reg_loss(self, _input:torch.Tensor):
        _, _, loss = self.vquantizer(_input)
        return loss
