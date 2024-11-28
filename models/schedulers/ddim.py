import torch

from models.schedulers.base import BaseScheduler


class DDIMScheduler(BaseScheduler):
    def __init__(self, n_diffusion_steps:int, n_backward_steps, beta_start:float=1e-4, beta_end:float=2e-2):
        super(DDIMScheduler, self).__init__(n_diffusion_steps, n_backward_steps, beta_start, beta_end)
    

    def backward(self, x:torch.Tensor, t:torch.LongTensor, eps:torch.Tensor) -> torch.Tensor:
        """
        Backward diffusion, based on DDIM:
        x_{t-delta} = alpha_{t-delta} * (x_t - eps * sqrt(1 - alpha_t)) / sqrt(alpha_t) + sqrt(1-alpha_{t-delta}) * eps

        Args:
            x (torch.Tensor): input tensor

            t (int): timestep

            eps (torch.Tensor): noise

        Returns:
            x_{t-delta} (torch.Tensor): sampled tensor
            t - delta (int): next time step
        """
        assert torch.all(t >= self.step_size), "Time step must be no less than step size"
        sqrt_alpha_t_minus_delta = self.sqrt_alpha_cumulative[t - self.step_size]
        sqrt_one_minus_alpha_t = self.sqrt_one_minus_alpha_cumulative[t]
        sqrt_alpha_t = self.sqrt_alpha_cumulative[t]
        sqrt_one_minus_alpha_t_minus_delta = self.sqrt_one_minus_alpha_cumulative[t - self.step_size]

        x_t_minus_delta = sqrt_alpha_t_minus_delta * (x - eps * sqrt_one_minus_alpha_t) / sqrt_alpha_t + sqrt_one_minus_alpha_t_minus_delta * eps

        return x_t_minus_delta, t - self.step_size
    