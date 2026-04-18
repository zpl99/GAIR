from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor


A1 = 1.340264
A2 = -0.081106
A3 = 0.000893
A4 = 0.003796
SF = 66.50336


def sample_b(sigma: float, size: tuple[int, int]) -> Tensor:
    return torch.randn(size) * sigma


@torch.jit.script
def gaussian_encoding(v: Tensor, b: Tensor) -> Tensor:
    vp = 2 * np.pi * v @ b.T
    return torch.cat((torch.cos(vp), torch.sin(vp)), dim=-1)


class GaussianEncoding(nn.Module):
    def __init__(
        self,
        sigma: Optional[float] = None,
        input_size: Optional[int] = None,
        encoded_size: Optional[int] = None,
        b: Optional[Tensor] = None,
    ) -> None:
        super().__init__()
        if b is None:
            if sigma is None or input_size is None or encoded_size is None:
                raise ValueError('Arguments "sigma", "input_size", and "encoded_size" are required.')
            b = sample_b(sigma, (encoded_size, input_size))
        elif sigma is not None or input_size is not None or encoded_size is not None:
            raise ValueError('Only specify the "b" argument when using it.')
        self.b = nn.parameter.Parameter(b, requires_grad=False)

    def forward(self, v: Tensor) -> Tensor:
        return gaussian_encoding(v, self.b)


def equal_earth_projection(coords: Tensor) -> Tensor:
    latitude = coords[:, 1]
    longitude = coords[:, 0]
    latitude_rad = torch.deg2rad(latitude)
    longitude_rad = torch.deg2rad(longitude)
    sqrt_three = torch.sqrt(torch.tensor(3.0, device=coords.device))
    sin_theta = (sqrt_three / 2) * torch.sin(latitude_rad)
    theta = torch.asin(sin_theta)
    denominator = 3 * (9 * A4 * theta**8 + 7 * A3 * theta**6 + 3 * A2 * theta**2 + A1)
    x = (2 * sqrt_three * longitude_rad * torch.cos(theta)) / denominator
    y = A4 * theta**9 + A3 * theta**7 + A2 * theta**3 + A1 * theta
    return (torch.stack((x, y), dim=1) * SF) / 180


class LocationEncoderCapsule(nn.Module):
    def __init__(self, sigma: float) -> None:
        super().__init__()
        rff_encoding = GaussianEncoding(sigma=sigma, input_size=2, encoded_size=256)
        self.capsule = nn.Sequential(
            rff_encoding,
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
        )
        self.head = nn.Sequential(nn.Linear(1024, 768))

    def forward(self, x: Tensor) -> Tensor:
        return self.head(self.capsule(x))


class LocationEncoder(nn.Module):
    def __init__(self, sigma: list[float] | tuple[float, ...] = (2**0, 2**4, 2**8), from_pretrained: bool = False) -> None:
        super().__init__()
        self.sigma = sigma
        self.n = len(self.sigma)
        for i, s in enumerate(self.sigma):
            self.add_module(f"LocEnc{i}", LocationEncoderCapsule(sigma=s))

    def forward(self, location: Tensor) -> Tensor:
        location = equal_earth_projection(location)
        features = torch.zeros(location.shape[0], 768, device=location.device, dtype=location.dtype)
        for i in range(self.n):
            features += self._modules[f"LocEnc{i}"](location)
        return features
