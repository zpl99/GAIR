from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import einsum, nn
from transformers import ViTConfig, ViTModel

from .croma import PretrainedCROMA
from .location_encoder import LocationEncoder
from .utils import calculate_relative_coordinates_normalized, get_2d_sincos_pos_embed, make_coord


DEFAULT_RS_INPUT_SIZE = 96
DEFAULT_SV_INPUT_SIZE = 224
DEFAULT_EMBED_DIM = 768


def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: torch.Tensor) -> torch.Tensor:
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, dtype=torch.float32, device=pos.device)
    omega /= embed_dim / 2.0
    omega = 1.0 / (10000**omega)
    pos = pos.reshape(-1)
    out = torch.einsum("m,d->md", pos, omega)
    emb_sin = torch.sin(out)
    emb_cos = torch.cos(out)
    return torch.cat([emb_sin, emb_cos], dim=1)


def pos_encoding_sin_cos(coords: torch.Tensor, embed_dim: int = DEFAULT_EMBED_DIM) -> torch.Tensor:
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, coords[:, 0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, coords[:, 1])
    return torch.cat([emb_h, emb_w], dim=1)


class FFN(nn.Module):
    def __init__(self, dim: int, mult: int = 4) -> None:
        super().__init__()
        inner_dim = int(dim * mult)
        self.net = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU(),
            nn.Linear(inner_dim, dim),
        )
        self.input_norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_norm(x)
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim: int, attention_heads: int = 8) -> None:
        super().__init__()
        self.attention_heads = attention_heads
        dim_head = int(dim / attention_heads)
        self.scale = dim_head**-0.5
        self.create_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.out = nn.Linear(dim, dim)
        self.input_norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        x = self.input_norm(x)
        q, k, v = self.create_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.attention_heads), (q, k, v))
        attention_scores = einsum("b h i d, b h j d -> b h i j", q, k) * self.scale
        attn = attention_scores.softmax(dim=-1)
        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        return self.out(rearrange(out, "b h n d -> b n (h d)") + pos)


class BaseTransformer(nn.Module):
    def __init__(self, dim: int, layers: int, attention_heads: int = 8, ff_mult: int = 4, final_norm: bool = True) -> None:
        super().__init__()
        self.final_norm = final_norm
        self.layers = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        Attention(dim=dim, attention_heads=attention_heads),
                        FFN(dim=dim, mult=ff_mult),
                    ]
                )
                for _ in range(layers)
            ]
        )
        if self.final_norm:
            self.norm_out = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, pos: torch.Tensor | None = None) -> torch.Tensor:
        for self_attn, ffn in self.layers:
            x = self_attn(x, pos) + x
            x = ffn(x) + x
        if self.final_norm:
            return self.norm_out(x)
        return x


class GAIRModel(nn.Module):
    def __init__(
        self,
        rs_input_size: int = DEFAULT_RS_INPUT_SIZE,
        sv_input_size: int = DEFAULT_SV_INPUT_SIZE,
        encoder_embed_dim: int = DEFAULT_EMBED_DIM,
        s2_channels: int = 10,
        queue_size: int = 4096,
        projection_input: int = DEFAULT_EMBED_DIM,
        projection_output: int = DEFAULT_EMBED_DIM,
        query_mode: str = "liif",
    ) -> None:
        super().__init__()
        self.rs_input_size = rs_input_size
        self.query_mode = query_mode

        self.rs_encoder = PretrainedCROMA(
            pretrained_path=None,
            size="base",
            modality="optical",
            image_resolution=rs_input_size,
            num_classes=10,
            s2_channels=s2_channels,
        )
        self.rs_patche_number = self.rs_encoder.num_patches**0.5

        config = ViTConfig(
            image_size=sv_input_size,
            patch_size=16,
            num_labels=1,
            hidden_size=encoder_embed_dim,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            hidden_act="gelu",
            layer_norm_eps=1e-12,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
        )
        self.sv_encoder = ViTModel(config)
        self.location_encoder = LocationEncoder(from_pretrained=False)

        self.pos_encoder = pos_encoding_sin_cos
        self.nerf_fn = BaseTransformer(encoder_embed_dim, layers=2)
        self.query_feature = nn.Parameter(torch.zeros(1, 1, encoder_embed_dim))
        self.rs_patch_pos_encoding = nn.Parameter(
            torch.zeros(1, self.rs_encoder.num_patches, encoder_embed_dim),
            requires_grad=False,
        )
        self.imnet = nn.Linear(encoder_embed_dim * 9 + 2, encoder_embed_dim)

        self.queue_size = queue_size
        self.register_buffer("queue_for_location", torch.randn(2, self.queue_size))
        self.queue_for_location = nn.functional.normalize(self.queue_for_location, dim=0)
        self.register_buffer("queue_ptr_location", torch.zeros(1, dtype=torch.long))

        self.radar_proj = nn.Linear(projection_input, projection_output)
        self.optical_proj = nn.Linear(projection_input, projection_output)
        self.location_proj = nn.Linear(projection_input, projection_output)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07), requires_grad=False)

        self.initialize_weights()

    def initialize_weights(self) -> None:
        patch_pos_encoding = get_2d_sincos_pos_embed(
            DEFAULT_EMBED_DIM,
            int(self.rs_encoder.num_patches**0.5),
            cls_token=False,
        )
        self.rs_patch_pos_encoding.data.copy_(torch.from_numpy(patch_pos_encoding).float().unsqueeze(0))
        torch.nn.init.normal_(self.query_feature, std=0.02)

    def freeze(self) -> "GAIRModel":
        self.eval()
        for param in self.parameters():
            param.requires_grad = False
        return self

    def load_checkpoint(self, checkpoint: str | Path | dict[str, Any], strict: bool = False) -> dict[str, Any]:
        if isinstance(checkpoint, (str, Path)):
            checkpoint_obj = torch.load(str(checkpoint), map_location="cpu")
        else:
            checkpoint_obj = checkpoint

        state_dict = checkpoint_obj["model"] if isinstance(checkpoint_obj, dict) and "model" in checkpoint_obj else checkpoint_obj
        if not isinstance(state_dict, dict):
            raise TypeError(f"Expected a state_dict-like object, got {type(state_dict)!r}")

        model_state = self.state_dict()
        filtered_state = {}
        shape_mismatch = {}

        for name, tensor in state_dict.items():
            clean_name = name[7:] if name.startswith("module.") else name
            if clean_name not in model_state:
                continue
            if model_state[clean_name].shape != tensor.shape:
                shape_mismatch[clean_name] = (tuple(model_state[clean_name].shape), tuple(tensor.shape))
                continue
            filtered_state[clean_name] = tensor

        incompatible = self.load_state_dict(filtered_state, strict=strict)
        return {
            "loaded": len(filtered_state),
            "missing_keys": list(incompatible.missing_keys),
            "unexpected_keys": list(incompatible.unexpected_keys),
            "shape_mismatch": shape_mismatch,
        }

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str | Path,
        *,
        device: str | torch.device = "cpu",
        query_mode: str = "liif",
        rs_input_size: int = DEFAULT_RS_INPUT_SIZE,
        sv_input_size: int = DEFAULT_SV_INPUT_SIZE,
    ) -> "GAIRModel":
        model = cls(rs_input_size=rs_input_size, sv_input_size=sv_input_size, query_mode=query_mode)
        load_info = model.load_checkpoint(checkpoint, strict=False)
        device = torch.device(device)
        model.to(device)
        model.freeze()
        model.load_info = load_info
        return model

    def encode_rs(self, rs_imgs: torch.Tensor, normalize: bool = False) -> torch.Tensor:
        _, rs_tokens = self.rs_encoder(rs_imgs)
        rs_embeddings = rs_tokens.mean(dim=1)
        return F.normalize(rs_embeddings, dim=1) if normalize else rs_embeddings

    def encode_sv(self, sv_imgs: torch.Tensor, normalize: bool = False) -> torch.Tensor:
        sv_embeddings = self.sv_encoder(sv_imgs)["pooler_output"]
        return F.normalize(sv_embeddings, dim=1) if normalize else sv_embeddings

    def encode_location(self, coordinates: torch.Tensor, normalize: bool = False) -> torch.Tensor:
        location_embeddings = self.location_encoder(coordinates)
        return F.normalize(location_embeddings, dim=1) if normalize else location_embeddings

    def forward(
        self,
        rs_imgs: torch.Tensor,
        sv_imgs: torch.Tensor,
        sv_coordinate: torch.Tensor | None = None,
        bbox_information: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.multi_model_forward(rs_imgs, sv_imgs, sv_coordinate, bbox_information)

    def multi_model_forward(
        self,
        rs_imgs: torch.Tensor,
        sv_imgs: torch.Tensor,
        sv_coordinate: torch.Tensor | None = None,
        bbox_information: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        rs_embeddings, queried_feature = self._query_feature(rs_imgs, sv_coordinate, bbox_information)
        sv_embeddings = self.encode_sv(sv_imgs, normalize=False)
        location_embeddings = self.encode_location(sv_coordinate, normalize=False) if sv_coordinate is not None else None
        return sv_embeddings, queried_feature.squeeze(1), location_embeddings, rs_embeddings

    def no_query(self, rs_imgs: torch.Tensor) -> torch.Tensor:
        return self.encode_rs(rs_imgs, normalize=False)

    def query_embedding(
        self,
        rs_imgs: torch.Tensor,
        sv_coordinate: torch.Tensor | None = None,
        bbox_information: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _, rs_tokens = self.rs_encoder(rs_imgs)
        sv_pos_relative_location = calculate_relative_coordinates_normalized(
            bbox_information,
            sv_coordinate,
            scale=self.rs_patche_number,
        )
        sv_pos_relative_embedding = self.pos_encoder(sv_pos_relative_location).unsqueeze(1)
        query_feature = self.query_feature.repeat(rs_tokens.shape[0], 1, 1)
        rs_info = torch.cat((rs_tokens, query_feature), dim=1)
        pos_info = torch.cat(
            [self.rs_patch_pos_encoding.repeat(rs_tokens.shape[0], 1, 1), sv_pos_relative_embedding],
            dim=1,
        )
        rs_nerf = self.nerf_fn(rs_info, pos_info)
        return rs_nerf[:, -1, :]

    def _query_feature(
        self,
        rs_imgs: torch.Tensor,
        sv_coordinate: torch.Tensor | None = None,
        bbox_information: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.query_mode == "liif":
            return self.liif_query_embedding(rs_imgs, sv_coordinate, bbox_information)
        if self.query_mode in ("bilinear", "bicubic"):
            return self.interp_query_embedding(rs_imgs, sv_coordinate, bbox_information, mode=self.query_mode)
        raise ValueError(f"Unsupported query_mode={self.query_mode!r}. Expected one of: liif, bilinear, bicubic.")

    def query_localized_rs(
        self,
        rs_imgs: torch.Tensor,
        coordinates: torch.Tensor,
        bboxes: torch.Tensor,
        *,
        mode: str | None = None,
        normalize: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        original_mode = self.query_mode
        if mode is not None:
            self.query_mode = mode
        try:
            rs_embeddings, localized = self._query_feature(rs_imgs, coordinates, bboxes)
        finally:
            self.query_mode = original_mode
        localized = localized.squeeze(1)
        if normalize:
            rs_embeddings = F.normalize(rs_embeddings, dim=1)
            localized = F.normalize(localized, dim=1)
        return rs_embeddings, localized

    def liif_query_embedding(
        self,
        rs_imgs: torch.Tensor,
        sv_coordinate: torch.Tensor | None = None,
        bbox_information: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        coord = calculate_relative_coordinates_normalized(bbox_information, sv_coordinate)
        coord = coord * 2 - 1
        coord = coord.unsqueeze(1)

        _, rs_tokens = self.rs_encoder(rs_imgs)
        rs_embeddings = rs_tokens.mean(1)
        batch_size, _, dim = rs_tokens.shape
        patch_num = int(self.rs_patche_number)
        rs_tokens = rs_tokens.view(batch_size, patch_num, patch_num, dim)
        feat = rs_tokens.permute(0, 3, 1, 2)
        feat = F.unfold(feat, 3, padding=1).view(feat.shape[0], feat.shape[1] * 9, feat.shape[2], feat.shape[3])

        vx_lst = [-1, 1]
        vy_lst = [-1, 1]
        eps_shift = 1e-6
        rx = 1 / feat.shape[-1]
        ry = 1 / feat.shape[-2]
        feat_coord = (
            make_coord(feat.shape[-2:], flatten=False)
            .to(feat.device)[..., [1, 0]]
            .permute(2, 0, 1)
            .unsqueeze(0)
            .expand(feat.shape[0], 2, *feat.shape[-2:])
        )

        preds = []
        areas = []
        for vx in vx_lst:
            for vy in vy_lst:
                coord_ = coord.clone()
                coord_[:, :, 0] += vx * rx + eps_shift
                coord_[:, :, 1] += vy * ry + eps_shift
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)

                q_feat = F.grid_sample(feat, coord_.unsqueeze(1), mode="nearest", align_corners=False)[:, :, 0, :].permute(0, 2, 1)
                q_coord = F.grid_sample(feat_coord, coord_.unsqueeze(1), mode="nearest", align_corners=False)[:, :, 0, :].permute(0, 2, 1)

                rel_coord = coord - q_coord
                rel_coord[:, :, 0] *= feat.shape[-1]
                rel_coord[:, :, 1] *= feat.shape[-2]
                inp = torch.cat([q_feat, rel_coord], dim=-1)

                bs, query_n = coord.shape[:2]
                pred = self.imnet(inp.view(bs * query_n, -1)).view(bs, query_n, -1)
                preds.append(pred)
                area = torch.abs(rel_coord[:, :, 0] * rel_coord[:, :, 1])
                areas.append(area + 1e-9)

        tot_area = torch.stack(areas).sum(dim=0)
        areas[0], areas[3] = areas[3], areas[0]
        areas[1], areas[2] = areas[2], areas[1]

        ret = 0
        for pred, area in zip(preds, areas):
            ret = ret + pred * (area / tot_area).unsqueeze(-1)
        return rs_embeddings, ret

    def interp_query_embedding(
        self,
        rs_imgs: torch.Tensor,
        sv_coordinate: torch.Tensor | None = None,
        bbox_information: torch.Tensor | None = None,
        mode: str = "bilinear",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if mode not in ("bilinear", "bicubic"):
            raise ValueError(f"Unsupported interpolation mode: {mode}")
        coord = calculate_relative_coordinates_normalized(bbox_information, sv_coordinate)
        coord = coord * 2 - 1
        coord = coord.unsqueeze(1)

        _, rs_tokens = self.rs_encoder(rs_imgs)
        rs_embeddings = rs_tokens.mean(1)
        batch_size, _, dim = rs_tokens.shape
        patch_num = int(self.rs_patche_number)
        rs_tokens = rs_tokens.view(batch_size, patch_num, patch_num, dim)
        feat = rs_tokens.permute(0, 3, 1, 2)
        queried_feature = F.grid_sample(feat, coord.unsqueeze(1), mode=mode, align_corners=False)[:, :, 0, :].permute(0, 2, 1)
        return rs_embeddings, queried_feature


RSSV_LIIF_style = GAIRModel


def vit_base_patch16_dec512d8b_liif(**kwargs: Any) -> GAIRModel:
    kwargs.setdefault("query_mode", "liif")
    return GAIRModel(**kwargs)


def vit_base_patch16_dec512d8b_liif_bilinear(**kwargs: Any) -> GAIRModel:
    kwargs.setdefault("query_mode", "bilinear")
    return GAIRModel(**kwargs)


def vit_base_patch16_dec512d8b_liif_bicubic(**kwargs: Any) -> GAIRModel:
    kwargs.setdefault("query_mode", "bicubic")
    return GAIRModel(**kwargs)
