from typing import Dict, List, Tuple
import torch
from torch import nn
import torch.nn.functional as F
import torchvision.transforms.v2.functional as TF
from PIL import Image
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, TimeElapsedColumn, MofNCompleteColumn
from torchvision.utils import make_grid
from diffusers import AutoencoderKL, AutoencoderTiny

from models.dit import DiffusionTransformer
from models.transformer.components import TextConditionEmbedding



class LDM(nn.Module):
    def __init__(self, vae:AutoencoderKL|AutoencoderTiny,
                 patch_size:int, channels:int, num_layers:int,
                 model_dim:int, hidden_dim, n_heads, 
                 dropout, activation, 
                 text_model_kwargs:List[Dict], token_limit:int, freeze_text_encoders:bool=True,
                 **activation_kwargs):
        super().__init__()
        print(activation_kwargs)
        self.autoencoder = vae
        self.diffusion = DiffusionTransformer(patch_size, channels, num_layers, model_dim, hidden_dim, n_heads, dropout, activation, **activation_kwargs)
        self.text_condition_embedding = TextConditionEmbedding(model_dim, text_model_kwargs, token_limit, freeze_text_encoders)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.to(self.device)
        self.inverse_scale_transform = True

        self.n_diffusion_steps = 1000
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

    def encode_text(self, text:str|list[str]):
        return self.text_condition_embedding(text)
        
    def forward(self, x:torch.Tensor, timesteps:torch.Tensor, c:torch.Tensor):
        return self.diffusion(x, timesteps, c)
    

    def forward_with_cfg(self, x, t, c, cfg_scale):
        return self.diffusion.forward_with_cfg(x, t, c, cfg_scale)
    
    @staticmethod
    def inverse_transform(tensors: torch.Tensor) -> torch.Tensor:
        """Convert tensors from [-1., 1.] to [0., 255.]"""
        return ((tensors.clamp(-1, 1) + 1.0) / 2.0) * 255.0

    @torch.no_grad()
    def backward_diffusion_sampling(self, condition_text:str|List[str], timesteps: int = 1000, 
                                    num_images: int = 1, return_grid=True, n_image_per_row: int = 5, dtype=torch.float32,
                                    image_size:int|Tuple[int]=(256, 256)) -> \
            Image.Image | List[Image.Image]:
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        print("Trying to get sample latent shape")

        dummy = torch.zeros((num_images, 3, *image_size), device=self.device, dtype=dtype)
        if isinstance(self.autoencoder, AutoencoderKL):
            shape = self.autoencoder.encode(dummy).latent_dist.sample().shape
        elif isinstance(self.autoencoder, AutoencoderTiny):
            shape = self.autoencoder.encode(dummy).latents.shape
        else:
            raise ValueError("Autoencoder must be an instance of AutoencoderKL or AutoencoderTiny")
        

        x = torch.randn((num_images, *shape[1:]), device=self.device, dtype=dtype)

        self.eval()

        pbar = Progress(TextColumn("Generating"), BarColumn(), MofNCompleteColumn(), TimeElapsedColumn(),
                        TimeRemainingColumn())

        task = pbar.add_task("", total=timesteps - 1)
        pbar.start()
        c = self.encode_text(condition_text)
        for time_step in range(timesteps - 1, 0, -1):
            ts = torch.full((num_images,), time_step, device=self.device, dtype=dtype)
            z = torch.randn_like(x, dtype=dtype) if time_step > 1 else torch.zeros_like(x, dtype=dtype)

            predicted_noise = self(x, ts, c)
            beta_t = self.betas[time_step]
            one_by_sqrt_alpha_t = self.one_by_sqrt_alpha[time_step]
            sqrt_one_minus_alpha_cumulative_t = self.sqrt_one_minus_alpha_cumulative[time_step]

            x = one_by_sqrt_alpha_t * (x - (beta_t / sqrt_one_minus_alpha_cumulative_t) * predicted_noise) \
                + torch.sqrt(beta_t) * z
            pbar.update(task, advance=1)
        x = self.autoencoder.decode(x).sample
        if self.inverse_scale_transform:
            x = self.inverse_transform(x).type(torch.uint8).to('cpu')
        else:
            x = x.to(dtype=torch.uint8).to('cpu')
        if return_grid:
            grid = make_grid(x, n_image_per_row)
            pil_image = TF.to_pil_image(grid)
            return pil_image
        returner = []
        for x_ in x:
            returner.append(TF.to_pil_image(x_))

        return returner