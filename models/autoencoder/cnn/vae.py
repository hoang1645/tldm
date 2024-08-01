import torch
from torch import nn
import torch.amp
from einops import rearrange
from .residual import EncoderResBlock, DecoderResBlock, MiddleBlock

from typing import List, Tuple, Dict


class VAE(nn.Module):
    def __init__(self, in_channels:int=3, latent_channels:int=4,
                 conv_channels:List[int]=[128, 256, 512, 512], encoder_block_num_res_blocks:int=2, decoder_block_num_res_blocks:int=3, groups:int=32):
        """a faithful recreation of VQ-VAE
        params:

        - `in_channels`: number of channels of the input (usually 3 (for RGB images))
        - `latent_channels`: number of channels of the latent space
        - `conv_channels`: a list of ints containing desired channels for each residual blocks in both encoder and decoder. The length of this list not only determines how many
        blocks each (encoder & decoder) will contain, but also the downsample factor: for `n` layers, the input will be downsized by `2 ^ (n - 1)` after forward propagation through
        the encoder.
        - `encoder_block_num_res_blocks`: an int, determining for each encoder block, how many residual blocks it will contain.
        - `decoder_block_num_res_blocks`: an int, determining for each decoder block, how many residual blocks it will contain.
        """
        nn.Module.__init__(self)
        self.device = 'cuda'
        conv_channels_rev = conv_channels[::-1]
        self.encoder_conv_init = nn.Conv2d(in_channels=in_channels, out_channels=conv_channels[0], kernel_size=7, padding='same')
        enc_res_blocks = []
        dec_res_blocks = []
        for i, (cin, cout) in enumerate(zip(conv_channels[:-1], conv_channels[1:])):
            enc_res_blocks.append(EncoderResBlock(cin, cout, encoder_block_num_res_blocks, groups=groups, downsample=(i < len(conv_channels) - 1)))
        for i, (cin, cout) in enumerate(zip(conv_channels_rev[:-1], conv_channels_rev[1:])):
            dec_res_blocks.append(DecoderResBlock(cin, cout, decoder_block_num_res_blocks, groups=groups, upsample=(i < len(conv_channels) - 1), attention=(i == 0)))
        self.encoder = nn.Sequential(*enc_res_blocks, MiddleBlock(conv_channels[-1]))
        self.mu_logvar = nn.Linear(conv_channels[-1], latent_channels * 2)
        

        self.decoder = nn.Sequential(nn.Conv2d(latent_channels, conv_channels[-1], 3, padding=1), *dec_res_blocks)
        self.out = nn.Conv2d(conv_channels[0], in_channels, 3, padding=1)

        self.to(self.device)


    def encode(self, x:torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.encoder_conv_init(x)
        x = self.encoder(x)

        x = rearrange(x, "b c h w -> b h w c")
        x = self.mu_logvar(x)
        x = rearrange(x, "b h w c -> b c h w")
        mu, logvar = torch.chunk(x, 2, dim=1)
        return {"mean": mu, "logvar": logvar}
        
    def reparameterize(self, mean:torch.Tensor, logvar:torch.Tensor)->torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std
    
    
    def kld(self, mean:torch.Tensor, logvar:torch.Tensor):
        z = self.reparameterize(mean, logvar)
        log_softmax = torch.nn.functional.log_softmax(z)
        target = torch.nn.functional.softmax(torch.randn_like(z))
        return torch.nn.functional.kl_div(log_softmax, target, reduction='batchmean')

    def decode(self, z:torch.Tensor) -> torch.Tensor:
        return self.out(self.decoder(z))
    
    def forward(self, x:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        result = self.encode(x)
        z = self.reparameterize(**result)
        kld = self.kld(**result)
        return self.decode(z), kld

