import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

import numpy as np
import math

def _cal_freq_list(freq_init, frequency_num, max_radius, min_radius):
    if freq_init == "random":
        # the frequence we use for each block, alpha in ICLR paper
        # freq_list shape: (frequency_num)
        freq_list = np.random.random(size=[frequency_num]) * max_radius
    elif freq_init == "geometric":
        # freq_list = []
        # for cur_freq in range(frequency_num):
        #     base = 1.0/(np.power(max_radius, cur_freq*1.0/(frequency_num-1)))
        #     freq_list.append(base)

        # freq_list = np.asarray(freq_list)

        log_timescale_increment = (math.log(float(max_radius) / float(min_radius)) /
          (frequency_num*1.0 - 1))

        timescales = min_radius * np.exp(
            np.arange(frequency_num).astype(float) * log_timescale_increment)

        freq_list = 1.0/timescales
    elif freq_init == "nerf":
        '''
        compute according to NeRF position encoding, 
        Equation 4 in https://arxiv.org/pdf/2003.08934.pdf 
        2^{0}*pi, ..., 2^{L-1}*pi
        '''
        #

        freq_list = np.pi * np.exp2(
            np.arange(frequency_num).astype(float))

    return freq_list


class SphereGridMixScaleSpatialRelationEncoder(nn.Module):
    """
    Given a list of (deltaX,deltaY), encode them using the position encoding function

    """

    def __init__(self, spa_embed_dim, coord_dim=2, frequency_num=16,
                 max_radius=10000, min_radius=10,
                 freq_init="geometric",
                 ffn=None):
        """
        Args:
            spa_embed_dim: the output spatial relation embedding dimention
            coord_dim: the dimention of space, 2D, 3D, or other
            frequency_num: the number of different sinusoidal with different frequencies/wavelengths
            max_radius: the largest context radius this model can handle
        """
        super(SphereGridMixScaleSpatialRelationEncoder, self).__init__()
        self.spa_embed_dim = spa_embed_dim
        self.coord_dim = coord_dim
        self.frequency_num = frequency_num
        self.freq_init = freq_init
        self.max_radius = max_radius
        self.min_radius = min_radius
        # the frequence we use for each block, alpha in ICLR paper
        self.cal_freq_list()
        self.cal_freq_mat()

        self.input_embed_dim = self.cal_input_dim()

        self.ffn = ffn
    def cal_elementwise_angle(self, coord, cur_freq):
        '''
        Args:
            coord: the deltaX or deltaY
            cur_freq: the frequency
        '''
        return coord / (np.power(self.max_radius, cur_freq * 1.0 / (self.frequency_num - 1)))

    def cal_coord_embed(self, coords_tuple):
        embed = []
        for coord in coords_tuple:
            for cur_freq in range(self.frequency_num):
                embed.append(math.sin(self.cal_elementwise_angle(coord, cur_freq)))
                embed.append(math.cos(self.cal_elementwise_angle(coord, cur_freq)))
        # embed: shape (input_embed_dim)
        return embed

    def cal_input_dim(self):
        # compute the dimention of the encoded spatial relation embedding
        return int(self.coord_dim * self.frequency_num * 2)

    def cal_freq_list(self):
        self.freq_list = _cal_freq_list(self.freq_init, self.frequency_num, self.max_radius, self.min_radius)

    def cal_freq_mat(self):
        # freq_mat shape: (frequency_num, 1)
        freq_mat = np.expand_dims(self.freq_list, axis=1)
        # self.freq_mat shape: (frequency_num, 1)
        self.freq_mat = freq_mat

    def make_input_embeds(self, coords):

        coords_mat = np.asarray(coords).astype(float)
        batch_size = coords_mat.shape[0]
        num_context_pt = coords_mat.shape[1]
        # coords_mat: shape (batch_size, num_context_pt, 2, 1)
        coords_mat = np.expand_dims(coords_mat, axis=3)
        # coords_mat: shape (batch_size, num_context_pt, 2, 1, 1)
        coords_mat = np.expand_dims(coords_mat, axis=4)
        # coords_mat: shape (batch_size, num_context_pt, 2, frequency_num, 1)
        coords_mat = np.repeat(coords_mat, self.frequency_num, axis=3)

        # convert to radius
        coords_mat = coords_mat * math.pi / 180

        # lon, lat: shape (batch_size, num_context_pt, 1, frequency_num, 1)
        lon_single = np.expand_dims(coords_mat[:, :, 0, :, :], axis=2)
        lat_single = np.expand_dims(coords_mat[:, :, 1, :, :], axis=2)

        # make sinuniod function
        # lon_sin, lon_cos: shape (batch_size, num_context_pt, 1, frequency_num, 1)
        lon_single_sin = np.sin(lon_single)
        lon_single_cos = np.cos(lon_single)

        # lat_sin, lat_cos: shape (batch_size, num_context_pt, 1, frequency_num, 1)
        lat_single_sin = np.sin(lat_single)
        lat_single_cos = np.cos(lat_single)

        # spr_embeds: shape (batch_size, num_context_pt, 2, frequency_num, 1)
        spr_embeds = coords_mat * self.freq_mat

        # lon, lat: shape (batch_size, num_context_pt, 1, frequency_num, 1)
        lon = np.expand_dims(spr_embeds[:, :, 0, :, :], axis=2)
        lat = np.expand_dims(spr_embeds[:, :, 1, :, :], axis=2)

        # make sinuniod function
        # lon_sin, lon_cos: shape (batch_size, num_context_pt, 1, frequency_num, 1)
        lon_sin = np.sin(lon)
        lon_cos = np.cos(lon)

        # lat_sin, lat_cos: shape (batch_size, num_context_pt, 1, frequency_num, 1)
        lat_sin = np.sin(lat)
        lat_cos = np.cos(lat)

        # spr_embeds_: shape (batch_size, num_context_pt, 1, frequency_num, 3)
        spr_embeds_ = np.concatenate(
            [lat_sin, lat_cos, lon_sin, lon_cos, lat_cos * lon_single_cos, lat_single_cos * lon_cos,
             lat_cos * lon_single_sin, lat_single_cos * lon_sin], axis=-1)

        spr_embeds = np.reshape(spr_embeds_, (batch_size, num_context_pt, -1))

        return spr_embeds

    def forward(self, coords,device):
        """
        Given a list of coords (deltaX, deltaY), give their spatial relation embedding
        Args:
            coords: a python list with shape (batch_size, num_context_pt, coord_dim)
        Return:
            sprenc: Tensor shape (batch_size, num_context_pt, spa_embed_dim)
        """

        spr_embeds = self.make_input_embeds(coords)

        spr_embeds = torch.FloatTensor(spr_embeds).to(device)


        if self.ffn is not None:
            return self.ffn(spr_embeds)
        else:
            return spr_embeds


