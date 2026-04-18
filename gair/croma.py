from __future__ import annotations

import itertools
import math

import torch
from einops import rearrange
from torch import einsum, nn


def get_2dalibi(num_heads: int, num_patches: int) -> torch.Tensor:
    points = list(itertools.product(range(int(math.sqrt(num_patches))), range(int(math.sqrt(num_patches)))))

    def get_slopes(n: int) -> list[float]:
        def get_slopes_power_of_2(m: int) -> list[float]:
            start = 2 ** (-2 ** -(math.log2(m) - 3))
            ratio = start
            return [start * ratio**i for i in range(m)]

        if math.log2(n).is_integer():
            return get_slopes_power_of_2(n)
        closest_power_of_2 = 2 ** math.floor(math.log2(n))
        return get_slopes_power_of_2(closest_power_of_2) + get_slopes(2 * closest_power_of_2)[0::2][: n - closest_power_of_2]

    slopes = torch.tensor(get_slopes(num_heads)).unsqueeze(1)
    idxs = []
    for p1 in points:
        for p2 in points:
            dist = math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
            idxs.append(dist * slopes * -1)
    all_bias = torch.cat(idxs, dim=1)
    return all_bias.view(1, num_heads, num_patches, num_patches)


class FFN(nn.Module):
    def __init__(self, dim: int, mult: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        inner_dim = int(dim * mult)
        self.net = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim),
        )
        self.input_norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_norm(x)
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.0) -> None:
        super().__init__()
        self.num_heads = num_heads
        dim_head = int(dim / num_heads)
        self.scale = dim_head**-0.5
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim)
        self.input_norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, relative_position_bias: torch.Tensor) -> torch.Tensor:
        x = self.input_norm(x)
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.num_heads), (q, k, v))
        attention_scores = einsum("b h i d, b h j d -> b h i j", q, k) * self.scale
        attention_scores = attention_scores + relative_position_bias
        attn = self.dropout(attention_scores.softmax(dim=-1))
        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class BaseTransformer(nn.Module):
    def __init__(self, dim: int, depth: int, num_heads: int = 8, ff_mult: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        Attention(dim=dim, num_heads=num_heads, dropout=dropout),
                        FFN(dim=dim, mult=ff_mult, dropout=dropout),
                    ]
                )
                for _ in range(depth)
            ]
        )

    def forward(self, x: torch.Tensor, relative_position_bias: torch.Tensor) -> torch.Tensor:
        for attn, ffn in self.layers:
            x = attn(x, relative_position_bias) + x
            x = ffn(x) + x
        return x


class ViT(nn.Module):
    def __init__(self, dim: int, depth: int, in_channels: int, num_heads: int = 16, patch_size: int = 8) -> None:
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        pixels_per_patch = patch_size * patch_size * in_channels
        self.linear_input = nn.Linear(pixels_per_patch, self.dim)
        self.transformer = BaseTransformer(dim=self.dim, depth=depth, num_heads=num_heads)

    def forward(self, imgs: torch.Tensor, attn_bias: torch.Tensor) -> torch.Tensor:
        x = rearrange(imgs, "b c (h i) (w j) -> b (h w) (c i j)", i=self.patch_size, j=self.patch_size)
        x = self.linear_input(x)
        return self.transformer(x, attn_bias)


class PretrainedCROMA(nn.Module):
    """Inference-only optical CROMA encoder.

    Despite the historic class name, this version does not load external weights.
    The GAIR checkpoint already contains all required RS-encoder parameters.
    """

    def __init__(
        self,
        pretrained_path: str | None = None,
        size: str = "base",
        modality: str = "optical",
        image_resolution: int = 96,
        num_classes: int = 768,
        s2_channels: int = 10,
    ) -> None:
        super().__init__()
        if size != "base":
            raise ValueError(f"Only size='base' is supported for GAIR inference, got {size!r}")
        if modality != "optical":
            raise ValueError(f"Only modality='optical' is supported for GAIR inference, got {modality!r}")
        if image_resolution % 8 != 0:
            raise ValueError(f"image_resolution must be a multiple of 8, got {image_resolution}")

        self.encoder_dim = 768
        self.encoder_depth = 12
        self.num_heads = 16
        self.patch_size = 8
        self.modality = modality
        self.num_patches = int((image_resolution / self.patch_size) ** 2)
        self.s2_channels = s2_channels
        self.num_classes = num_classes

        self.register_buffer("attn_bias", get_2dalibi(num_heads=self.num_heads, num_patches=self.num_patches), persistent=False)
        self.s2_encoder = ViT(dim=self.encoder_dim, depth=self.encoder_depth, in_channels=self.s2_channels, num_heads=self.num_heads)
        self.layer_norm = nn.LayerNorm(self.encoder_dim)
        self.head = nn.Linear(self.encoder_dim, self.num_classes)

    def freeze_parameters(self) -> None:
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, optical_images: torch.Tensor, return_hidden_state: bool = False):
        optical_encodings = self.s2_encoder(imgs=optical_images, attn_bias=self.attn_bias.to(optical_images.device))
        optical_gap = self.head(self.layer_norm(optical_encodings.mean(dim=1)))
        if return_hidden_state:
            return optical_gap, optical_encodings, []
        return optical_gap, optical_encodings
