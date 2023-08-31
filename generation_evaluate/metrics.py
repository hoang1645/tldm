import torch
from torch import nn
from torchvision.models import inception_v3, Inception_V3_Weights
import scipy.linalg
import numpy as np


class FID(object):
    def __init__(self, fp16: bool = False):
        self.inception_model = inception_v3(Inception_V3_Weights.IMAGENET1K_V1)
        self.inception_model.fc = nn.Identity()
        self.inception_model.eval()
        if fp16:
            self.inception_model.half()
        self.fp16 = fp16
        self.reals = []
        self.fakes = []

    def append(self, reals: torch.Tensor|None=None, fakes: torch.Tensor|None=None):
        if reals is not None:
            reals = self._preprocess(reals)
            self.reals.append(self.inception_model(reals.half() if self.fp16 else reals).cpu())
        if fakes is not None:
            fakes = self._preprocess(fakes)
            self.fakes.append(self.inception_model(fakes.half() if self.fp16 else fakes).cpu())

    @staticmethod
    @torch.no_grad()
    def _matrix_sqrt(x: torch.Tensor) -> torch.Tensor:
        return torch.Tensor(scipy.linalg.sqrtm(x.cpu().detach().numpy()))

    @staticmethod
    @torch.no_grad()
    def _frechet_distance(mu_x: torch.Tensor, mu_y: torch.Tensor,
                          sigma_x: torch.Tensor, sigma_y: torch.Tensor) -> torch.Tensor:
        return (mu_x - mu_y).dot(mu_x - mu_y) \
            + torch.trace(sigma_x) + torch.trace(sigma_y) - 2 * torch.trace(FID._matrix_sqrt(sigma_x @ sigma_y))

    @staticmethod
    @torch.no_grad()
    def _preprocess(img: torch.Tensor):
        return torch.nn.functional.interpolate(img, size=(299, 299), mode='bilinear', align_corners=False)

    @staticmethod
    @torch.no_grad()
    def _get_covariance(features: torch.Tensor) -> torch.Tensor:
        return torch.Tensor(np.cov(features.detach().numpy(), rowvar=False))

    @torch.no_grad()
    def frechet_inception_distance(self):
        fake_features_all = torch.cat(self.fakes)
        real_features_all = torch.cat(self.reals)
        mu_fake = fake_features_all.mean(0)
        mu_real = real_features_all.mean(0)
        sigma_fake = self._get_covariance(fake_features_all)
        sigma_real = self._get_covariance(real_features_all)

        return self._frechet_distance(mu_fake, mu_real, sigma_fake, sigma_real).item()

