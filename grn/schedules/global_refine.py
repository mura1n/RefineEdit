import os
import json
import math
import bisect

import numpy as np
import torch
import torch.nn.functional as F

from grn.utils_t2iv.hbq_util_t2iv import multiclass_labels2onehot_input


def shift_pt(pt, alpha):
    """shift pt (signal ratio) to lower one, recommand alpha=sqrt(height*width/256/256)"""
    if alpha > 1000:
        alpha = alpha - 1000
    noise_pt = 1 - pt
    noise_pt = alpha * noise_pt / (1+(alpha-1)*noise_pt)
    pt = 1 - noise_pt
    return pt


def get_visual_rope_embeds(rope2d_freqs_grid, scale_schedule, device=None, mapped_h_div_w_template=None, t_offset=0):

    rope2d_freqs_grid['freqs_frames'] = rope2d_freqs_grid['freqs_frames'].to(device)
    rope2d_freqs_grid['freqs_height'] = rope2d_freqs_grid['freqs_height'].to(device)
    rope2d_freqs_grid['freqs_width'] = rope2d_freqs_grid['freqs_width'].to(device)
    max_height = rope2d_freqs_grid['freqs_height'].shape[1]
    max_width = rope2d_freqs_grid['freqs_width'].shape[1]
    extreme_h_div_w = 3
    assert mapped_h_div_w_template <= extreme_h_div_w
    extreme_h = max_height
    extreme_w = extreme_h / extreme_h_div_w
    upw = np.sqrt(extreme_h * extreme_w / mapped_h_div_w_template)
    uph = mapped_h_div_w_template * upw
    uph, upw = int(uph), int(upw)
    pt, ph, pw = scale_schedule
    assert ph <= uph and pw <= upw
    f_frames = rope2d_freqs_grid['freqs_frames'][:, t_offset:t_offset+pt]
    f_height = rope2d_freqs_grid['freqs_height'][:, (torch.arange(ph) * (uph / ph)).round().int()]
    f_width = rope2d_freqs_grid['freqs_width'][:, (torch.arange(pw) * (upw / pw)).round().int()]
    rope_embeds = torch.cat([
        f_frames[   :,     :,  None,   None,   :].expand(-1, -1, ph, pw, -1),
        f_height[   :,  None,      :,  None,   :].expand(-1,  pt,-1, pw, -1),
        f_width[   :,  None,   None,      :,   :].expand(-1,  pt,ph, -1, -1),
    ], dim=-1)
    rope_embeds = rope_embeds.reshape(2, 1, 1, 1, pt*ph*pw, -1)
    return rope_embeds
