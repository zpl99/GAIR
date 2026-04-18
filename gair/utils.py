from __future__ import annotations

import numpy as np
import torch


def calculate_relative_coordinates_normalized(
    bboxes: torch.Tensor,
    coords: torch.Tensor,
    scale: float = 1.0,
) -> torch.Tensor:
    """Return relative query coordinates inside each bbox.

    `bboxes` must use the order `[lon_min, lat_max, lon_max, lat_min]`.
    `coords` must use the order `[lon, lat]`.
    """
    lon_min, lat_max, lon_max, lat_min = (
        bboxes[:, 0],
        bboxes[:, 1],
        bboxes[:, 2],
        bboxes[:, 3],
    )
    lon = coords[:, 0]
    lat = coords[:, 1]

    x_relative = (lon - lon_min) / (lon_max - lon_min)
    y_relative = (lat_max - lat) / (lat_max - lat_min)
    relative_coords = torch.stack((x_relative, y_relative), dim=-1)
    return relative_coords * scale


def get_2d_sincos_pos_embed(
    embed_dim: int,
    grid_size: int,
    cls_token: bool = False,
    len_extra_tokens: int = 1,
) -> np.ndarray:
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0).reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([len_extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim: int, grid: np.ndarray) -> np.ndarray:
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1)


def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega

    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    return np.concatenate([emb_sin, emb_cos], axis=1)


def make_coord(shape: tuple[int, int], flatten: bool = True) -> torch.Tensor:
    """Make coordinates at grid centers in [-1, 1]."""
    coord_seqs = []
    for n in shape:
        v0, v1 = -1.0, 1.0
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs, indexing="ij"), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret
