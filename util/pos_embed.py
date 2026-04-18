# --------------------------------------------------------
# References:
# MAE: https://github.com/facebookresearch/mae
# --------------------------------------------------------

import numpy as np
import torch
import rasterio
# --------------------------------------------------------
# 2D sine-cosine position embedding
# References:
# Transformer: https://github.com/tensorflow/models/blob/master/official/nlp/transformer/model_utils.py
# MoCo v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------


def calculate_relative_coordinates_normalized(a,b, scale=1):

    lon_min, lat_max, lon_max, lat_min = a[:, 0], a[:, 1], a[:, 2], a[:, 3]

    lat = b[:, 1]
    lon = b[:, 0]

    x_relative = (lon - lon_min) / (lon_max - lon_min)
    y_relative = (lat_max - lat) / (lat_max - lat_min)

    relative_coords = torch.stack((x_relative, y_relative), dim=-1)

    return relative_coords*scale

def calculate_relative_coordinates_scaled(a, b, scale):


    lon_min, lat_max, lon_max, lat_min = a[:, 0], a[:, 1], a[:, 2], a[:, 3]

    lat = b[:, :, 1]
    lon = b[:, :, 0]

    x_relative = (lon - lon_min.unsqueeze(1)) / (lon_max.unsqueeze(1) - lon_min.unsqueeze(1))
    y_relative = (lat_max.unsqueeze(1) - lat) / (lat_max.unsqueeze(1) - lat_min.unsqueeze(1))

    if isinstance(scale, (float,int)):
        x_relative = x_relative * scale - 0.5
        y_relative = y_relative * scale - 0.5
    elif isinstance(scale,(list,tuple)):
        x_relative = x_relative * scale[0] - 0.5
        y_relative = y_relative * scale[1] - 0.5

    relative_coords = torch.stack((x_relative, y_relative), dim=-1)

    return relative_coords

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False,len_extra_tokens=1):


    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([len_extra_tokens, embed_dim]), pos_embed], axis=0)

    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):

    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def get_1d_sincos_pos_embed_from_grid_torch(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, dtype=float, device=pos.device)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = torch.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = torch.sin(out) # (M, D/2)
    emb_cos = torch.cos(out) # (M, D/2)

    emb = torch.cat([emb_sin, emb_cos], dim=1)  # (M, D)
    return emb.double()


def get_batch_2d_sincos_pos_embed_from_lonlat(embed_dim, lat_lon_coords):
    """
    lat_lon_coords: Tensor or numpy array with shape (N, L, 2) where N=batch size, L=number of patches, 2=lat and lon
    embed_dim: The embedding dimension, should be divisible by 2
    return:
    pos_embed: [N, L, embed_dim] 2D sine-cosine positional encoding for lat/lon coordinates
    """
    assert embed_dim % 2 == 0, "Embedding dimension should be divisible by 2"

    # Separate lat and lon from lat_lon_coords
    lat = lat_lon_coords[..., 0]  # shape (N, L)
    lon = lat_lon_coords[..., 1]  # shape (N, L)
    # Get the sine-cosine positional encoding for lat and lon
    lat_embed = get_batch_1d_sincos_pos_embed_from_lonlat(embed_dim // 2, lat)  # shape (N, L, D/2)
    lon_embed = get_batch_1d_sincos_pos_embed_from_lonlat(embed_dim // 2, lon)  # shape (N, L, D/2)

    # Concatenate latitude and longitude embeddings along the last dimension
    pos_embed = torch.concat([lat_embed, lon_embed], dim=-1)  # shape (N, L, D)

    return pos_embed


def get_batch_1d_sincos_pos_embed_from_lonlat(embed_dim, pos):
    """
    embed_dim: output dimension for each position (should be D/2 for lat or lon)
    pos: a batch of positions to be encoded, shape (N, L)
    out: (N, L, D)
    """
    assert embed_dim % 2 == 0, "Embedding dimension should be divisible by 2"

    N, L = pos.shape  # Get batch size (N) and number of positions (L)

    # Create omega using PyTorch (D/2,)
    omega = torch.arange(embed_dim // 2, dtype=torch.float32, device=pos.device)
    omega /= embed_dim
    omega = 1. / (10000 ** omega)  # shape (D/2,)

    # Ensure pos is shaped (N, L)
    pos = pos.view(N, L)  # pos shape is already (N, L) by default

    # Perform outer product between pos and omega
    out = torch.einsum('nl,d->nld', pos, omega)  # shape (N, L, D/2), batch outer product

    # Compute sine and cosine embeddings
    emb_sin = torch.sin(out)  # Sine embedding (N, L, D/2)
    emb_cos = torch.cos(out)  # Cosine embedding (N, L, D/2)

    # Concatenate sine and cosine embeddings along the last dimension
    emb = torch.cat([emb_sin, emb_cos], dim=-1)  # shape (N, L, D)
    return emb

# --------------------------------------------------------
# Interpolate position embeddings for high-resolution
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------
def interpolate_pos_embed(model, checkpoint_model):
    if 'pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        try:
            num_patches = model.patch_embed.num_patches
        except AttributeError as err:
            num_patches = model.patch_embed[0].num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model['pos_embed'] = new_pos_embed
