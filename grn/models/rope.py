import math
import os
from functools import partial
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from timm.models.layers import DropPath, drop_path
from torch.utils.checkpoint import checkpoint


def precompute_rope3d_freqs_grid(
        dim,
        rope2d_normalized_by_hw,
        max_frames=128,
        max_height=2048 // 8,
        max_width=2048 // 8,
        base=10000.0,
        device=None,
        activated_h_div_w_templates=[],
        text_maxlen=0,
        pn=None,
        args=None,
        **kwargs,
):

    print(f'[precompute_rope4d_freqs_grid: 3d]: start')
    assert dim % 2 == 0, f'Only support dim % 2 == 0, but got dim={dim}'
    dim_div_2 = dim // 2
    num_of_freqs_former = dim_div_2 // 3
    preserve_1d_length = 600
    num_of_freqs_last = dim_div_2 - num_of_freqs_former * 2
    inv_freq_former = 1.0 / (base ** (torch.arange(num_of_freqs_former, dtype=torch.int64).float().to(device) / num_of_freqs_former))
    inv_freq_last = 1.0 / (base ** (torch.arange(num_of_freqs_last, dtype=torch.int64).float().to(device) / num_of_freqs_last))
    t_frames = torch.arange(preserve_1d_length+max_frames, device=device, dtype=torch.int64).type_as(inv_freq_former)
    t_height = torch.arange(max_height, device=device, dtype=torch.int64).type_as(inv_freq_former)
    t_width = torch.arange(max_width, device=device, dtype=torch.int64).type_as(inv_freq_former)
    freqs_frames = torch.outer(t_frames, inv_freq_former)
    freqs_height = torch.outer(t_height, inv_freq_former)
    freqs_width = torch.outer(t_width, inv_freq_last)
    freqs_frames = torch.stack([torch.cos(freqs_frames), torch.sin(freqs_frames)], dim=0)
    freqs_height = torch.stack([torch.cos(freqs_height), torch.sin(freqs_height)], dim=0)
    freqs_width = torch.stack([torch.cos(freqs_width), torch.sin(freqs_width)], dim=0)
    tm = preserve_1d_length
    rope_text_embeds = torch.cat([
        freqs_frames[   :,   :tm,  None,   None,   :].expand(-1, -1, -1, -1, -1),
        freqs_height[   :,  None,    :1,   None,   :].expand(-1, tm, -1, -1, -1),
        freqs_width[   :,  None,  None,     :1,   :].expand(-1, tm, -1, -1, -1),
    ], dim=-1)
    rope_text_embeds = rope_text_embeds.reshape(2, 1, 1, 1, tm, dim_div_2)
    rope2d_freqs_grid = {}
    rope2d_freqs_grid['freqs_text'] = rope_text_embeds
    rope2d_freqs_grid['freqs_frames'] = freqs_frames[:, tm:]
    rope2d_freqs_grid['freqs_height'] = freqs_height
    rope2d_freqs_grid['freqs_width'] = freqs_width
    return rope2d_freqs_grid


def apply_rotary_emb(q, k, rope_cache):
    device_type = q.device.type
    device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
    qk = [q, k]
    rope_cache = rope_cache[:,0]
    with torch.autocast(device_type=device_type, enabled=False):
        for i in range(2):
            qk[i] = qk[i].reshape(*qk[i].shape[:-1], -1, 2)
            tmp1 = qk[i][..., 1] * rope_cache[1]
            tmp2 = qk[i][..., 0] * rope_cache[1]
            qk[i][..., 0].mul_(rope_cache[0]).sub_(tmp1)
            qk[i][..., 1].mul_(rope_cache[0]).add_(tmp2)
            qk[i] = qk[i].reshape(*qk[i].shape[:-2], -1)
        q, k = qk


    return q, k
