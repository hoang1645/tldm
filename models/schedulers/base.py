import torch

from abc import ABC, abstractmethod


class BaseScheduler(ABC):
    def __init__(self, n_diffusion_steps:int, n_backward_steps:int, beta_start:float=1e-4, beta_end:float=2e-2):
        assert n_diffusion_steps % n_backward_steps == 0, "n_diffusion_steps must be divisible by n_backward_steps"

        self.n_diffusion_steps = n_diffusion_steps
        self.n_backward_steps = n_backward_steps

        self.step_size = n_diffusion_steps // n_backward_steps


        self.betas = self.__get_betas(n_diffusion_steps, beta_start, beta_end)
        self.alphas = 1 - self.betas
        self.sqrt_beta = torch.sqrt(self.betas)
        self.alpha_cumulative = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_cumulative = torch.sqrt(self.alpha_cumulative)
        self.one_by_sqrt_alpha = 1. / torch.sqrt(self.alphas)
        self.sqrt_one_minus_alpha_cumulative = torch.sqrt(1 - self.alpha_cumulative)

    def __get_betas(self, n_diffusion_steps, beta_start, beta_end) -> torch.Tensor:
        return torch.linspace(
            beta_start,
            beta_end,
            n_diffusion_steps
        )

    @abstractmethod
    def forward(self, x:torch.Tensor, timesteps:torch.Tensor):
        """
        Perform forward diffusion (based on DDPM):
        x_t = sqrt(alpha_t) * x0 + sqrt(1 - alpha_t) * N(0, I)

        Args:
            x (torch.Tensor): input tensor

            timesteps (torch.Tensor): timesteps to sample from

        Returns:
            x_t (torch.Tensor): sampled tensor

            eps (torch.Tensor): noise
        """
        eps = torch.randn_like(x)
        mean = self.sqrt_alpha_cumulative[timesteps].reshape(-1, 1, 1, 1) * x
        std = self.sqrt_one_minus_alpha_cumulative[timesteps].reshape(-1, 1, 1, 1)

        sample = eps * std + mean
        return sample, eps

    @abstractmethod
    def backward(self, x, t):
        raise NotImplementedError
    

