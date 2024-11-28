import torch

from models.schedulers.base import BaseScheduler


class DDPMScheduler(BaseScheduler):
    def __init__(self, n_diffusion_steps:int, beta_start:float=1e-4, beta_end:float=2e-2):
        super(DDPMScheduler, self).__init__(n_diffusion_steps, n_diffusion_steps, beta_start, beta_end)
    

    def backward(self, x:torch.Tensor, t:torch.LongTensor, eps:torch.Tensor) -> torch.Tensor:
        """
        Backward diffusion, based on DDPM:
        x_{t-1} = 1/sqrt(alpha_t) * (x_t - beta_t/sqrt(1 - alpha_t) + beta_t * eps

        Args:
            x (torch.Tensor): input tensor

            t (int): timestep

            eps (torch.Tensor): noise

        Returns:
            x_{t-1} (torch.Tensor): sampled tensor
            t - 1 (int): next time step
        """

        x_t_minus_1 = self.one_by_sqrt_alpha[t] * (x - self.betas[t].to(x.device) * self.sqrt_one_minus_alpha_cumulative[t].to(x.device) * eps) \
            + (z := torch.randn_like(x, dtype=x.dtype) if t > 1 else torch.zeros_like(x, dtype=x.dtype)).to(x.device) * self.sqrt_beta[t].to(x.device)
        return x_t_minus_1, t - 1
    