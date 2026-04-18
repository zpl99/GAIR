import torch

import torch.nn.functional as F
from torch import distributed as dist
from torch import nn, einsum
from einops import rearrange
from util.pos_embed import calculate_relative_coordinates_normalized, get_2d_sincos_pos_embed
import numpy as np

from croma import use_croma
import copy
from nif.utils import make_coord
from location_encoder.location_encoder import LocationEncoder
from transformers import ViTModel, ViTConfig


def apply_mask_to_sequence(x, ids_keep):
    return torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, x.shape[-1]))


def exists(val):
    return val is not None


class FFN(nn.Module):
    def __init__(self,
                 dim,
                 mult=4,
                 ):
        super().__init__()
        inner_dim = int(dim * mult)
        self.net = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU(),
            nn.Linear(inner_dim, dim)
        )
        self.input_norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.input_norm(x)
        return self.net(x)


class Attention(nn.Module):
    def __init__(self,
                 dim,
                 attention_heads=8,
                 ):
        super().__init__()
        self.attention_heads = attention_heads
        dim_head = int(dim / attention_heads)
        self.scale = dim_head ** -0.5
        self.create_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.out = nn.Linear(dim, dim)
        self.input_norm = nn.LayerNorm(dim)

    def forward(self, x, pos):
        x = self.input_norm(x)
        q, k, v = self.create_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.attention_heads), (q, k, v))
        attention_scores = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = attention_scores.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        return self.out(rearrange(out, 'b h n d -> b n (h d)') + pos)


class BaseTransformer(nn.Module):
    def __init__(self,
                 dim,
                 layers,
                 attention_heads=8,
                 ff_mult=4,
                 final_norm=True,
                 ):
        super().__init__()
        self.final_norm = final_norm
        self.layers = nn.ModuleList([])
        for _ in range(layers):
            self.layers.append(nn.ModuleList([
                Attention(dim=dim, attention_heads=attention_heads),
                FFN(dim=dim, mult=ff_mult),
            ]))
        if self.final_norm:
            self.norm_out = nn.LayerNorm(dim)

    def forward(self, x, pos=None):
        for self_attn, ffn in self.layers:
            x = self_attn(x, pos) + x
            x = ffn(x) + x
        if self.final_norm:
            return self.norm_out(x)
        else:
            return x


class ViT(nn.Module):
    def __init__(self,
                 num_patches,
                 dim=768,
                 layers=12,
                 attention_heads=16,
                 in_channels=12,
                 patch_size=8,
                 ):
        super().__init__()
        self.dim = dim
        self.layers = layers
        self.attention_heads = attention_heads
        self.num_patches = num_patches
        self.patch_size = patch_size
        pixels_per_patch = int(patch_size * patch_size * in_channels)
        self.linear_input = nn.Linear(pixels_per_patch, self.dim)
        self.transformer = BaseTransformer(dim=self.dim,
                                           layers=self.layers,
                                           attention_heads=self.attention_heads,
                                           )

    def forward(self, imgs, attn_bias, mask_info=None):
        x = rearrange(imgs, 'b c (h i) (w j) -> b (h w) (c i j)', i=self.patch_size, j=self.patch_size)
        x = self.linear_input(x)
        if mask_info is None:
            x = self.transformer(x, alibi=attn_bias)
            return x
        else:
            x_masked = apply_mask_to_sequence(x=x, ids_keep=mask_info['ids_keep'])
            x_masked = self.transformer(x_masked, alibi=attn_bias)
            return x_masked


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, dtype=torch.float32).to(embed_dim.device)
    omega /= embed_dim / 2.
    omega = 1. / (10000 ** omega)  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = torch.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = torch.sin(out)  # (M, D/2)
    emb_cos = torch.cos(out)  # (M, D/2)

    emb = torch.cat([emb_sin, emb_cos], dim=1)  # (M, D)
    return emb


def pos_encoding_sin_cos(coords, embed_dim=768):
    assert embed_dim % 2 == 0
    embed_dim = torch.tensor(embed_dim).to(coords.device)
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, coords[:, 0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, coords[:, 1])  # (H*W, D/2)

    emb = torch.cat([emb_h, emb_w], dim=1)  # (H*W, D)
    return emb

class RSSV_LIIF_style(nn.Module):
    def __init__(self, rs_input_size=128, sv_input_size=224, encoder_embed_dim=768,
                 s2_channels=10, queue_size=4096, projection_input=768, projection_output=768):
        super(RSSV_LIIF_style, self).__init__()

        self.rs_input_size = rs_input_size

        self.rs_encoder = use_croma.PretrainedCROMA(
            pretrained_path='./CROMA_base.pt',
            size='base', modality='optical', image_resolution=rs_input_size, s2_channels=s2_channels)

        self.rs_encoder_teacher = copy.deepcopy(self.rs_encoder)

        self.rs_patche_number = self.rs_encoder.num_patches ** .5



        config = ViTConfig(
            image_size=sv_input_size,
            patch_size=16,
            num_labels=1,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            hidden_act="gelu",
            layer_norm_eps=1e-12,
            dropout_rate=0.1,
            attention_probs_dropout_prob=0.1
        )


        self.sv_encoder = ViTModel(config)

        self.sv_encoder_teacher = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')

        self.location_encoder = LocationEncoder(from_pretrained=False)


        self.pos_encoder = pos_encoding_sin_cos
        self.nerf_fn = BaseTransformer(encoder_embed_dim, layers=2)
        self.ContrastLoss = ContrastLossInput()
        self.query_feature = nn.Parameter(torch.zeros(1, 1, encoder_embed_dim))
        self.rs_patch_pos_encoding = nn.Parameter(torch.zeros(1, self.rs_encoder.num_patches, encoder_embed_dim),
                                                  requires_grad=False)

        # LIFF part
        self.imnet = nn.Linear(encoder_embed_dim * 9 + 2, encoder_embed_dim)
        # create the queue for pos encoding
        self.queue_size = queue_size
        self.register_buffer("queue_for_location", torch.randn(2, self.queue_size))
        self.register_buffer("queue_for_rs_encoding", torch.randn(768, self.queue_size))
        self.queue_for_location = nn.functional.normalize(self.queue_for_location, dim=0)
        self.queue_for_rs_encoding = nn.functional.normalize(self.queue_for_rs_encoding, dim=0)
        self.register_buffer("queue_ptr_location", torch.zeros(1, dtype=torch.long))
        self.register_buffer("queue_ptr_rs", torch.zeros(1, dtype=torch.long))

        # loss function part
        self.radar_proj = nn.Linear(projection_input, projection_output)
        self.optical_proj = nn.Linear(projection_input, projection_output)
        self.location_proj = nn.Linear(projection_input, projection_output)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07), requires_grad=False)

        # teacher part
        self.cos = nn.CosineSimilarity(dim=1)
        self.teacher_avgpool = torch.nn.AdaptiveAvgPool1d(1)
        self.teacher_alpha = 0.2
        # student part
        self.rs_student_projector = nn.Linear(768,768)
        self.sv_student_projector = nn.Linear(768,768)

        self.initialize_weights()

    def forward_loss(self, input, target):
        cv_loss = self.ContrastLoss(input, target)
        return cv_loss


    def cross_attention(self, queried_feature, sv_patches):
        pass

    @torch.no_grad()
    def dequeue_and_enqueue_loc(self, location_encoding):
        location_encoding = concat_all_gather(location_encoding)

        batch_size = location_encoding.shape[0]

        ptr_loc = int(self.queue_ptr_location)

        assert self.queue_size % batch_size == 0  # for simplicity

        self.queue_for_location[:, ptr_loc: ptr_loc + batch_size] = location_encoding.T

        ptr_loc = (ptr_loc + batch_size) % self.queue_size
        self.queue_ptr_location[0] = ptr_loc

    @torch.no_grad()
    def dequeue_and_enqueue_rs(self, rs_feature_encoding):
        rs_feature_encoding = concat_all_gather(rs_feature_encoding)

        batch_size = rs_feature_encoding.shape[0]

        ptr_rs = int(self.queue_ptr_rs)

        assert self.queue_size % batch_size == 0  # for simplicity

        self.queue_for_rs_encoding[:, ptr_rs: ptr_rs + batch_size] = rs_feature_encoding.T

        ptr_rs = (ptr_rs + batch_size) % self.queue_size
        self.queue_ptr_rs[0] = ptr_rs

    def moco_style_loss(self, pos, neg, temperature=0.07):
        # logits: Nx(1+K)
        logits = torch.cat([pos, neg], dim=1)
        # apply temperature
        logits /= temperature
        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        return F.cross_entropy(logits, labels)

    def contrastive_loss(self, radar_features, optical_features, location_features_pos, location_features_neg):
        # linear projection of unimodal representations
        # radar_features = self.radar_proj(radar_features)
        optical_features = self.optical_proj(optical_features)
        # location_features = self.location_proj(location_features)

        # L2 normalize
        radar_features = radar_features / radar_features.norm(dim=1, keepdim=True)
        optical_features = optical_features / optical_features.norm(dim=1, keepdim=True)
        location_features_pos = location_features_pos / location_features_pos.norm(dim=1, keepdim=True)
        location_features_neg = location_features_neg / location_features_neg.norm(dim=1, keepdim=True)
        # location_features = location_features/location_features.norm(dim=1,keepdim=True)

        # dot product to get logits
        logit_scale = self.logit_scale.exp()

        # sv and loc
        l_pos_sv_loc = torch.einsum("nc,nc->n", [optical_features, location_features_pos]).unsqueeze(-1)
        l_pos_rs_loc = torch.einsum("nc,nc->n", [radar_features, location_features_pos]).unsqueeze(-1)

        # negative logits: NxK
        l_neg_sv_loc = torch.einsum("nc,ck->nk", [optical_features, location_features_neg.t()])
        l_neg_rs_sv = torch.einsum("nc,ck->nk", [radar_features, location_features_neg.t()])

        sv_loc_loss = self.moco_style_loss(l_pos_sv_loc, l_neg_sv_loc)
        rs_loc_loss = self.moco_style_loss(l_pos_rs_loc, l_neg_rs_sv)

        logits_per_optical = logit_scale * optical_features @ radar_features.t()

        logits_per_radar = logit_scale * radar_features @ optical_features.t()

        # organize labels
        num_logits = logits_per_optical.shape[0]
        labels = torch.arange(num_logits, device=radar_features.device, dtype=torch.long)

        # calculate loss
        loss = (F.cross_entropy(logits_per_optical, labels) + F.cross_entropy(logits_per_radar, labels) + sv_loc_loss + rs_loc_loss) / 2
        return loss

    def forward(self, rs_imgs, sv_imgs, sv_coordinate=None, bbox_information=None):
        rs_embeddings, queried_feature = self.liif_query_embedding(rs_imgs, sv_coordinate, bbox_information)
        sv_embeddings_ts = self.forward_sv_teacher(sv_imgs)
        rs_embeddings_ts = self.forward_rs_teacher(rs_imgs)
        queried_feature = queried_feature.squeeze(1)  # [B,768]
        # queried_feature = self.query_embedding(rs_imgs, sv_coordinate, bbox_information)

        # sv_embeddings, sv_tokens = self.sv_encoder.forward_features(sv_imgs, return_patch_embeddings=True)  # [B,768], for training from scrach
        sv_embeddings = self.sv_encoder(sv_imgs)['pooler_output'] # [B,768]
        sv_coordinate_neg = self.queue_for_location.t()
        # sv_coordinate_all = torch.cat([sv_coordinate,sv_coordinate_neg],dim=0)
        location_embeddings_pos = self.location_encoder(sv_coordinate)  # [B,768]
        location_embeddings_neg = self.location_encoder(sv_coordinate_neg)

        loss = self.contrastive_loss(queried_feature, sv_embeddings, location_embeddings_pos, location_embeddings_neg)
        self.queue_process(sv_coordinate)

        distillation_loss = self.distillation_loss(sv_embeddings,rs_embeddings,sv_embeddings_ts,rs_embeddings_ts)
        return loss+distillation_loss

    def forward_sv_teacher(self,sv_imgs):
        with torch.no_grad():
            sv_teacher_output = self.sv_encoder_teacher(sv_imgs)
            sv_teacher_feature = sv_teacher_output['pooler_output']
            # sv_teacher_feature=torch.flatten(sv_teacher_feature, 1)
        return sv_teacher_feature

    def forward_rs_teacher(self,rs_imgs):
        with torch.no_grad():
            _, rs_teacher_output = self.rs_encoder_teacher(rs_imgs)
            rs_teacher_output = self.teacher_avgpool(rs_teacher_output.transpose(1, 2))
            rs_teacher_feature = torch.flatten(rs_teacher_output, 1)
        return rs_teacher_feature

    def distillation_loss(self, sv_student_feature, rs_student_feature, sv_teacher_feature, rs_teacher_feature):
        distill_loss1 = -(self.cos(sv_student_feature, sv_teacher_feature.detach()).mean()) * self.teacher_alpha
        distill_loss2 = -(self.cos(rs_student_feature, rs_teacher_feature.detach()).mean()) * self.teacher_alpha
        return distill_loss1 + distill_loss2

    def queue_process(self, loc=None, rs_embeddings=None):
        if loc is not None:
            self.dequeue_and_enqueue_loc(loc)
        if rs_embeddings is not None:
            self.dequeue_and_enqueue_rs(rs_embeddings)

    def initialize_weights(self):
        patch_pos_encoding = get_2d_sincos_pos_embed(768, int(self.rs_encoder.num_patches ** .5), cls_token=False)
        self.rs_patch_pos_encoding.data.copy_(torch.from_numpy(patch_pos_encoding).float().unsqueeze(0))
        torch.nn.init.normal_(self.query_feature, std=.02)
        # torch.nn.init.normal_(self.mask_token, std=.02)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def multi_model_forward(self, rs_imgs, sv_imgs, sv_coordinate=None, bbox_information=None):
        rs_embeddings, queried_feature = self.liif_query_embedding(rs_imgs, sv_coordinate, bbox_information)
        queried_feature = queried_feature.squeeze(1)  # [B,768]

        sv_embeddings = self.sv_encoder(sv_imgs)['pooler_output']  # [B,768]
        location_embeddings = self.location_encoder(sv_coordinate)

        return sv_embeddings,queried_feature,location_embeddings,rs_embeddings

    def query_embedding(self, rs_imgs, sv_coordinate=None, bbox_information=None):
        _, rs_tokens = self.rs_encoder(rs_imgs)  # [B,L, 768]
        sv_pos_relative_location = calculate_relative_coordinates_normalized(bbox_information, sv_coordinate,
                                                                             scale=self.rs_patche_number)  # [B,2]
        sv_pos_relative_embedding = self.pos_encoder(sv_pos_relative_location)  # [B,768]
        sv_pos_relative_embedding = sv_pos_relative_embedding.unsqueeze(1)  # [B, 1, 768]
        query_feature = self.query_feature.repeat(rs_tokens.shape[0], 1, 1)
        rs_info = torch.cat((rs_tokens, query_feature), dim=1)  # [B, L+1, 768]
        pos_info = torch.cat([self.rs_patch_pos_encoding.repeat(rs_tokens.shape[0], 1, 1), sv_pos_relative_embedding],
                             dim=1)
        rs_nerf = self.nerf_fn(rs_info, pos_info)  # [B, L+1, 768]
        queried_feature = rs_nerf[:, -1, :]  # [B,768]
        return queried_feature

    def liif_query_embedding(self, rs_imgs, sv_coordinate=None, bbox_information=None):
        coord = calculate_relative_coordinates_normalized(bbox_information, sv_coordinate)
        coord = coord.unsqueeze(1)
        _, rs_tokens = self.rs_encoder(rs_imgs)  # [B,L, 768]
        rs_embeddings = rs_tokens.mean(1) # [B,768]
        B, L, dim = rs_tokens.shape
        rs_tokens = rs_tokens.view(B, int(self.rs_patche_number), int(self.rs_patche_number), dim)
        feat = rs_tokens.permute(0, 3, 1, 2)  # [B,768,h,w]
        feat = F.unfold(feat, 3, padding=1).view(
            feat.shape[0], feat.shape[1] * 9, feat.shape[2], feat.shape[3])
        vx_lst = [-1, 1]
        vy_lst = [-1, 1]
        eps_shift = 1e-6
        rx = 2 / feat.shape[-2] / 2
        ry = 2 / feat.shape[-1] / 2
        feat_coord = make_coord(feat.shape[-2:], flatten=False).cuda() \
            .permute(2, 0, 1) \
            .unsqueeze(0).expand(feat.shape[0], 2, *feat.shape[-2:])
        preds = []
        areas = []
        for vx in vx_lst:
            for vy in vy_lst:
                coord_ = coord.clone()
                coord_[:, :, 0] += vx * rx + eps_shift
                coord_[:, :, 1] += vy * ry + eps_shift
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)
                q_feat = F.grid_sample(
                    feat, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                q_coord = F.grid_sample(
                    feat_coord, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                rel_coord = coord - q_coord
                rel_coord[:, :, 0] *= feat.shape[-2]
                rel_coord[:, :, 1] *= feat.shape[-1]
                inp = torch.cat([q_feat, rel_coord], dim=-1)

                bs, q = coord.shape[:2]
                pred = self.imnet(inp.view(bs * q, -1)).view(bs, q, -1)
                preds.append(pred)

                area = torch.abs(rel_coord[:, :, 0] * rel_coord[:, :, 1])
                areas.append(area + 1e-9)

        tot_area = torch.stack(areas).sum(dim=0)
        t = areas[0];
        areas[0] = areas[3];
        areas[3] = t
        t = areas[1];
        areas[1] = areas[2];
        areas[2] = t

        ret = 0
        for pred, area in zip(preds, areas):
            ret = ret + pred * (area / tot_area).unsqueeze(-1)
        return rs_embeddings, ret

def gather_features(features, world_size):
    gathered_image_features = [torch.zeros_like(features) for _ in range(world_size)]
    dist.all_gather(gathered_image_features, features)
    all_features = torch.cat(gathered_image_features, dim=0)
    return all_features


class CrossAttention(nn.Module):
    def __init__(self,
                 dim,
                 attention_heads=8,
                 ):
        super().__init__()
        self.attention_heads = attention_heads
        dim_head = int(dim / attention_heads)
        self.scale = dim_head ** -0.5
        self.create_q = nn.Linear(dim, dim, bias=False)
        self.create_k = nn.Linear(dim, dim, bias=False)
        self.create_v = nn.Linear(dim, dim, bias=False)
        self.to_out = nn.Linear(dim, dim)
        self.input_norm = nn.LayerNorm(dim)

    def forward(self, x, context, alibi=None):
        x = self.input_norm(x)
        context = self.input_norm(context)
        q = self.create_q(x)
        k = self.create_k(context)
        v = self.create_v(context)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.attention_heads), (q, k, v))
        attention_scores = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attention_scores = attention_scores  # + alibi
        attn = attention_scores.softmax(dim=-1)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class BaseTransformerCrossAttn(nn.Module):
    def __init__(self,
                 dim,
                 layers,
                 attention_heads=8,
                 ff_mult=4,
                 ):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(layers):
            self.layers.append(nn.ModuleList([
                Attention(dim=dim, attention_heads=attention_heads),
                CrossAttention(dim=dim, attention_heads=attention_heads),
                FFN(dim=dim, mult=ff_mult),
            ]))
        self.norm_out = nn.LayerNorm(dim)

    def forward(self, x, context, alibi):
        for self_attn, cross_attn, ffn in self.layers:
            x = self_attn(x, alibi) + x
            x = cross_attn(x, context, alibi) + x
            x = ffn(x) + x
        x = self.norm_out(x)
        return x


class ContrastLossInput(nn.Module):
    def __init__(
            self,
            projection_input=768,
            projection_output=768,
    ):
        super().__init__()
        self.radar_proj = nn.Linear(projection_input, projection_output)
        self.optical_proj = nn.Linear(projection_input, projection_output)
        self.location_proj = nn.Linear(projection_input, projection_output)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07), requires_grad=False)

    def forward(self, radar_features, optical_features, location_features):
        # linear projection of unimodal representations
        radar_features = self.radar_proj(radar_features)
        optical_features = self.optical_proj(optical_features)
        location_features = self.location_proj(location_features)

        # L2 normalize
        radar_features = radar_features / radar_features.norm(dim=1, keepdim=True)
        optical_features = optical_features / optical_features.norm(dim=1, keepdim=True)
        location_features = location_features / location_features.norm(dim=1, keepdim=True)
        # dot product to get logits
        logit_scale = self.logit_scale.exp()
        logits_per_optical = logit_scale * optical_features @ radar_features.t()
        logits_per_radar = logit_scale * radar_features @ optical_features.t()

        # organize labels
        num_logits = logits_per_optical.shape[0]
        labels = torch.arange(num_logits, device=radar_features.device, dtype=torch.long)

        # calculate loss
        loss = (F.cross_entropy(logits_per_optical, labels) + F.cross_entropy(logits_per_radar, labels)) / 2
        return loss



def vit_base_patch16_dec512d8b_liif(**kwargs):
    model = RSSV_LIIF_style(**kwargs)
    return model

# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [
        torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output










