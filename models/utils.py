from torch import nn 

Upsample = lambda dim, dim_out : nn.Sequential(nn.Upsample(scale_factor=2), 
                                               nn.Conv2d(dim, dim_out if dim_out is not None else dim, 
                                                                                      kernel_size=(3, 3), padding='same'))

Downsample = lambda dim, dim_out: nn.Sequential(nn.PixelUnshuffle(2),
                                                nn.Conv2d(dim << 2, dim_out if dim_out is not None else dim, 
                                                                                      kernel_size=(1, 1)))