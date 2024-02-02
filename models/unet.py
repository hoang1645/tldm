import torch
from torch import nn
from .embeddings import  TimestepEmbedding
from .cnn import ResnetBlock
from .attention import FlashMultiHeadAttn
from .utils import Downsample, Upsample
from typing import Tuple


class UNet(nn.Module):
    def __init__(self, in_chan:int=3, out_chan:int=3, 
                 embed_dim:int=32, conv_init_chan:int=32,
                 chan_mults:Tuple[int, int, int, int]=(1, 2, 3, 4),
                 norm_groups:int=8, n_attn_heads=4, dim_head=32,
                 where_attn:Tuple[bool, bool, bool, bool]=(False, False, False, True),
                 ):
        super().__init__()
        # initial conv layer
        self.init_conv = nn.Conv2d(in_chan, conv_init_chan, kernel_size=(7, 7), padding='same')
        # time embedding
        self.time_embedding = nn.Sequential(TimestepEmbedding(embed_dim),
                                            nn.Linear(embed_dim, embed_dim<<2), nn.GELU(),
                                            nn.Linear(embed_dim<<2, embed_dim<<2))
        # flag to indicate last down block
        is_last_down_block = [False] * (len(chan_mults) - 1) + [True]
        # define unet down blocks
        self.down_residual = nn.ModuleList([])
        self.down_attn = nn.ModuleList([])
        self.downsamplers = nn.ModuleList([])
        chan_mults_before = [1] + list(chan_mults)
        
        for chan_mult_prev, chan_mult_curr, is_attn, is_last_block in \
        zip(chan_mults_before[:-1], chan_mults_before[1:], where_attn, is_last_down_block):
            # residual block
            self.down_residual.append(residual:=nn.ModuleList([ResnetBlock(conv_init_chan * chan_mult_prev,
                                                                          conv_init_chan * chan_mult_prev,
                                                                          time_emb_dim=embed_dim<<2, groups=norm_groups),
                                                                ResnetBlock(conv_init_chan * chan_mult_prev,
                                                                          conv_init_chan * chan_mult_prev,
                                                                          time_emb_dim=embed_dim<<2, groups=norm_groups),
                                                                          ]))
            # attention block
            self.down_attn.append(FlashMultiHeadAttn(conv_init_chan * chan_mult_prev, n_attn_heads, dim_head)
                                  if is_attn else nn.Identity())
            # downsampler
            self.downsamplers.append(Downsample(conv_init_chan * chan_mult_prev, conv_init_chan * chan_mult_curr) 
                                     if not is_last_block 
                                     else nn.Conv2d(conv_init_chan * chan_mult_prev, conv_init_chan * chan_mult_curr, 3, padding = 1))
            
        # bottleneck blocks
        self.bottleneck_conv1 = ResnetBlock(conv_init_chan * chan_mults[-1], conv_init_chan * chan_mults[-1],
                                            time_emb_dim=embed_dim << 2, groups=norm_groups)
        self.bottleneck_attn = FlashMultiHeadAttn(conv_init_chan * chan_mults[-1], n_attn_heads, dim_head)
        self.bottleneck_conv2 = ResnetBlock(conv_init_chan * chan_mults[-1], conv_init_chan * chan_mults[-1],
                                            time_emb_dim=embed_dim << 2, groups=norm_groups)
        
        # up blocks
        self.up_residual = nn.ModuleList([])
        self.up_attn = nn.ModuleList([])
        self.upsamplers = nn.ModuleList([])
        for chan_mult_prev, chan_mult_curr, is_attn, is_last_block in \
        zip(chan_mults_before[:0:-1], chan_mults_before[-2::-1], where_attn[::-1], is_last_down_block[::-1]):
            # residual block
            self.up_residual.append(residual:=nn.ModuleList([ResnetBlock(conv_init_chan * (chan_mult_prev << 1),
                                                                          conv_init_chan * chan_mult_prev,
                                                                          time_emb_dim=embed_dim<<2, groups=norm_groups),
                                                                ResnetBlock(conv_init_chan * chan_mult_prev,
                                                                          conv_init_chan * chan_mult_prev,
                                                                          time_emb_dim=embed_dim<<2, groups=norm_groups),
                                                                ResnetBlock(conv_init_chan * chan_mult_prev,
                                                                          conv_init_chan * chan_mult_prev,
                                                                          time_emb_dim=embed_dim<<2, groups=norm_groups),
                                                                          ]))
            # attention block
            self.up_attn.append(FlashMultiHeadAttn(conv_init_chan * chan_mult_prev, n_attn_heads, dim_head)
                                  if is_attn else nn.Identity())
            # upsampler
            self.upsamplers.append(Upsample(conv_init_chan * chan_mult_prev, conv_init_chan * chan_mult_curr) 
                                     if not is_last_block 
                                     else nn.Conv2d(conv_init_chan * chan_mult_prev, conv_init_chan * chan_mult_curr, 3, padding = 1))
            
            self.final_res_block = ResnetBlock(conv_init_chan * 2, conv_init_chan, time_emb_dim = embed_dim<<2, groups=norm_groups)
            self.last_conv = nn.Conv2d(conv_init_chan, out_chan, kernel_size=(3, 3), padding='same')

    def forward(self, x:torch.Tensor, time:torch.Tensor):
        time_emb = self.time_embedding(time)
        # print(time_emb.shape)
        x = self.init_conv(x)
        r = x # saved for concatenation

        # going down
        h = [] # saving down blocks outputs for concatenation
        for down_blocks, down_attn, downsampler in zip(self.down_residual, self.down_attn, self.downsamplers):
            for down_block in down_blocks:
                x = down_block(x, time_emb)
            x = down_attn(x)
            x = downsampler(x)
            h.append(x)
        
        # bottleneck
        x = self.bottleneck_conv1(x, time_emb)
        x = self.bottleneck_attn(x)
        x = self.bottleneck_conv2(x, time_emb)

        # going up
        for up_blocks, up_attn, upsampler in zip(self.up_residual, self.up_attn, self.upsamplers):
            r0 = h.pop()
            # print(r0.shape, x.shape)
            x = torch.cat([x, r0], dim=1)
            for up_block in up_blocks: x = up_block(x, time_emb)
            x = up_attn(x)
            x = upsampler(x)
        
        # finalize
        x = torch.cat((r, x), dim=1)
        x = self.final_res_block(x, time_emb)
        x = self.last_conv(x)
        return x
        

    


# def cast_tuple(t, length = 1):
#     if isinstance(t, tuple):
#         return t
#     return ((t,) * length)

# class Unet(nn.Module):
#     def __init__(
#         self,
#         dim,
#         init_dim = None,
#         out_dim = None,
#         dim_mults = (1, 2, 4, 8),
#         channels = 3,
#         self_condition = False,
#         resnet_block_groups = 8,
#         learned_variance = False,
#         learned_sinusoidal_cond = False,
#         random_fourier_features = False,
#         learned_sinusoidal_dim = 16,
#         attn_dim_head = 32,
#         attn_heads = 4,
#         full_attn = (False, False, False, True),
#         # flash_attn = False
#     ):
#         super().__init__()

#         # determine dimensions

#         self.channels = channels
#         self.self_condition = self_condition
#         input_channels = channels * (2 if self_condition else 1)

#         if init_dim is None: init_dim = dim 
#         self.init_conv = nn.Conv2d(input_channels, init_dim, 7, padding = 3)

#         dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
#         in_out = list(zip(dims[:-1], dims[1:]))

#         block_klass = partial(ResnetBlock, groups = resnet_block_groups)

#         # time embeddings

#         time_dim = dim * 4

#         self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

#         if self.random_or_learned_sinusoidal_cond:
#             sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
#             fourier_dim = learned_sinusoidal_dim + 1
#         else:
#             sinu_pos_emb = SinusoidalPosEmb(dim)
#             fourier_dim = dim

#         self.time_mlp = nn.Sequential(
#             sinu_pos_emb,
#             nn.Linear(fourier_dim, time_dim),
#             nn.GELU(),
#             nn.Linear(time_dim, time_dim)
#         )

#         # attention

#         num_stages = len(dim_mults)
#         full_attn  = cast_tuple(full_attn, num_stages)
#         attn_heads = cast_tuple(attn_heads, num_stages)
#         attn_dim_head = cast_tuple(attn_dim_head, num_stages)

#         assert len(full_attn) == len(dim_mults)

#         FullAttention = FlashMultiHeadAttn

#         # layers

#         self.downs = nn.ModuleList([])
#         self.ups = nn.ModuleList([])
#         num_resolutions = len(in_out)

#         for ind, ((dim_in, dim_out), layer_full_attn, layer_attn_heads, layer_attn_dim_head) in enumerate(zip(in_out, full_attn, attn_heads, attn_dim_head)):
#             is_last = ind >= (num_resolutions - 1)

#             attn_klass = FullAttention if layer_full_attn else LinearAttention

#             self.downs.append(nn.ModuleList([
#                 block_klass(dim_in, dim_in, time_emb_dim = time_dim),
#                 block_klass(dim_in, dim_in, time_emb_dim = time_dim),
#                 attn_klass(dim_in, dim_head = layer_attn_dim_head, heads = layer_attn_heads),
#                 Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding = 1)
#             ]))

#         mid_dim = dims[-1]
#         self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim)
#         self.mid_attn = FullAttention(mid_dim, heads = attn_heads[-1], dim_head = attn_dim_head[-1])
#         self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim)

#         for ind, ((dim_in, dim_out), layer_full_attn, layer_attn_heads, layer_attn_dim_head) in enumerate(zip(*map(reversed, (in_out, full_attn, attn_heads, attn_dim_head)))):
#             is_last = ind == (len(in_out) - 1)

#             attn_klass = FullAttention if layer_full_attn else LinearAttention

#             self.ups.append(nn.ModuleList([
#                 block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim),
#                 block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim),
#                 attn_klass(dim_out, dim_head = layer_attn_dim_head, heads = layer_attn_heads),
#                 Upsample(dim_out, dim_in) if not is_last else  nn.Conv2d(dim_out, dim_in, 3, padding = 1)
#             ]))

#         default_out_dim = channels * (1 if not learned_variance else 2)
#         self.out_dim = out_dim if out_dim is not None else default_out_dim

#         self.final_res_block = block_klass(dim * 2, dim, time_emb_dim = time_dim)
#         self.final_conv = nn.Conv2d(dim, self.out_dim, 1)

#     @property
#     def downsample_factor(self):
#         return 2 ** (len(self.downs) - 1)

#     def forward(self, x, time, x_self_cond = None):
#         assert all([d % self.downsample_factor == 0 for d in x.shape[-2:]]), f'your input dimensions {x.shape[-2:]} need to be divisible by {self.downsample_factor}, given the unet'

#         if self.self_condition:
#             if x_self_cond is None:  x_self_cond = lambda: torch.zeros_like(x)
#             x = torch.cat((x_self_cond, x), dim = 1)

#         x = self.init_conv(x)
#         r = x.clone()

#         t = self.time_mlp(time)

#         h = []

#         for block1, block2, attn, downsample in self.downs:
#             x = block1(x, t)
#             h.append(x)

#             x = block2(x, t)
#             x = attn(x) + x
#             h.append(x)

#             x = downsample(x)

#         x = self.mid_block1(x, t)
#         x = self.mid_attn(x) + x
#         x = self.mid_block2(x, t)

#         for block1, block2, attn, upsample in self.ups:
#             x = torch.cat((x, h.pop()), dim = 1)
#             x = block1(x, t)

#             x = torch.cat((x, h.pop()), dim = 1)
#             x = block2(x, t)
#             x = attn(x) + x

#             x = upsample(x)

#         x = torch.cat((x, r), dim = 1)

#         x = self.final_res_block(x, t)
#         return self.final_conv(x)
