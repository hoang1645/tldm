import torch
from models.schedulers.base import BaseScheduler

    
class EulerDiscreteScheduler(BaseScheduler):
    def __init__(self, n_diffusion_steps, n_backward_steps,
                 beta_start = 0.00085, beta_end = 0.012):
        super().__init__(n_diffusion_steps, n_backward_steps, beta_start, beta_end)
        self.sigmas = (self.sqrt_one_minus_alpha_cumulative / self.sqrt_alpha_cumulative) # sigma schedule


    def forward(self, x:torch.Tensor, t:torch.LongTensor):
        sigmas = self.sigmas.to(x.device)
        sigma = sigmas[t]
        noise = torch.randn_like(x)
        x_t = x + sigma * noise
        return x_t, noise

    def backward(self, x:torch.Tensor, t:torch.LongTensor, eps:torch.Tensor, s_churn: float = 0.0,
        s_tmin: float = 0.0,
        s_tmax: float = float("inf"),
        s_noise: float = 1.0,
) -> torch.Tensor:
        assert torch.all(t >= self.step_size), "Time step must be no less than step size"
        sigma = self.sigmas[t]

        gamma = 0
        if sigma > s_tmin and sigma < s_tmax:
            gamma = min(s_churn / (len(self.sigmas) + 1), 2 ** .5 - 1)

        sigma_hat = sigma + gamma * sigma

        if gamma > 0:
            noise = torch.randn_like(eps)
            eps_prime = noise * s_noise
            x = x + eps_prime * (sigma_hat**2 - sigma**2) ** 0.5 # line 6

        pred_original_sample = x - sigma_hat * eps
        d = (x - pred_original_sample) / (sigma_hat + 1e-8)
        
        dt = self.sigmas[(t - self.step_size).clamp(0, len(self.sigmas) - 1)] - sigma_hat

        x_t_minus_delta = x + d * dt

        return x_t_minus_delta, t - self.step_size
        