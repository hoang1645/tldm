import torch
from torch import nn
from .unet import UNet
from .autoencoder.autoencoders import AutoencoderVQ
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, TimeElapsedColumn, MofNCompleteColumn
import torchvision.transforms.v2.functional as TF
from torchvision.utils import make_grid
from time import time
from PIL import Image
from typing import List


class LDM(nn.Module):
    def __init__(self, unet: UNet, autoencoder: AutoencoderVQ,
                 image_size: int, image_channels: int = 32, n_diffusion_steps: int = 1000,
                 device: str | torch.device = 'cuda', inverse_scale_transform: bool = True):
        super().__init__()

        self.unet = unet
        self.autoencoder = autoencoder
        self.image_dim = (image_channels, image_size // 8, image_size // 8)
        self.n_diffusion_steps = n_diffusion_steps
        self.device = device
        self.inverse_scale_transform = inverse_scale_transform
        self.to(device)

        # initialize
        self.betas = self.__get_betas()
        self.alphas = 1 - self.betas
        self.sqrt_beta = torch.sqrt(self.betas)
        self.alpha_cumulative = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_cumulative = torch.sqrt(self.alpha_cumulative)
        self.one_by_sqrt_alpha = 1. / torch.sqrt(self.alphas)
        self.sqrt_one_minus_alpha_cumulative = torch.sqrt(1 - self.alpha_cumulative)

    def __get_betas(self, scheduler: str = 'linear') -> torch.Tensor:
        if scheduler == 'linear':
            scale = 1000 / self.n_diffusion_steps
            beta_start = scale * 1e-4
            beta_end = scale * 0.02
            return torch.linspace(
                beta_start,
                beta_end,
                self.n_diffusion_steps,
                device=self.device
            )
        raise NotImplementedError("Beta schedulers other than linear are not yet implemented")

    def forward_diffusion(self, x0: torch.Tensor, timesteps: int | torch.Tensor):
        eps = torch.randn_like(x0)
        mean = self.sqrt_alpha_cumulative[timesteps].reshape(-1, 1, 1, 1) * x0
        std = self.sqrt_one_minus_alpha_cumulative[timesteps].reshape(-1, 1, 1, 1)

        sample = eps * std + mean
        return sample, eps

    def forward(self, x0: torch.Tensor, timesteps: torch.Tensor):
        return self.unet(x0, timesteps)

    @staticmethod
    def inverse_transform(tensors: torch.Tensor) -> torch.Tensor:
        """Convert tensors from [-1., 1.] to [0., 255.]"""
        return ((tensors.clamp(-1, 1) + 1.0) / 2.0) * 255.0

    @torch.no_grad()
    def backward_diffusion_sampling(self, timesteps: int = 1000, num_images: int = 1, return_grid=True, n_image_per_row: int = 5, dtype=torch.float32) -> \
            Image.Image | List[Image.Image]:
        x = torch.randn((num_images, 4, 16, 16), device=self.device, dtype=dtype)
        self.eval()

        pbar = Progress(TextColumn("Generating"), BarColumn(), MofNCompleteColumn(), TimeElapsedColumn(),
                        TimeRemainingColumn())

        task = pbar.add_task("", total=timesteps - 1)
        pbar.start()
        for time_step in range(timesteps - 1, 0, -1):
            ts = torch.full((num_images,), time_step, device=self.device, dtype=dtype)
            z = torch.randn_like(x, dtype=dtype) if time_step > 1 else torch.zeros_like(x, dtype=dtype)

            predicted_noise = self(x, ts)
            beta_t = self.betas[time_step]
            one_by_sqrt_alpha_t = self.one_by_sqrt_alpha[time_step]
            sqrt_one_minus_alpha_cumulative_t = self.sqrt_one_minus_alpha_cumulative[time_step]

            x = one_by_sqrt_alpha_t * (x - (beta_t / sqrt_one_minus_alpha_cumulative_t) * predicted_noise) \
                + torch.sqrt(beta_t) * z
            pbar.update(task, advance=1)
        x = self.autoencoder.decode(x)
        if self.inverse_scale_transform:
            x = self.inverse_transform(x).type(torch.uint8).to('cpu')
        else:
            x = x.to(dtype=torch.uint8).to('cpu')
        if return_grid:
            grid = make_grid(x, n_image_per_row)
            pil_image = TF.to_pil_image(grid)
            # pil_image.save(f'generated_{time()}.jpg')
            return pil_image
        returner = []
        for x_ in x:
            returner.append(TF.to_pil_image(x_))

        return returner

