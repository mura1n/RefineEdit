import json
import math
import copy

import tqdm
import numpy as np


def get_first_full_spatial_size_scale_index(vae_scale_schedule):
    for si, (pt, ph, pw) in enumerate(vae_scale_schedule):
        if vae_scale_schedule[si][-2:] == vae_scale_schedule[-1][-2:]:
            return si


def get_ratio2hws_pixels2scales(dynamic_scale_schedule, train_h_div_w_list, video_frames):
    compressed_frames = video_frames // 4 + 1
    if dynamic_scale_schedule in ['GRN_vae_stride16']:
        assert type(train_h_div_w_list) is str
        train_h_div_w_list = json.loads(train_h_div_w_list)
        if len(train_h_div_w_list) == 0:
            train_h_div_w_list = [3/1, 5/2, 2/1, 16/9, 3/2, 4/3, 116/100, 1, 100/116, 3/4, 2/3, 9/16, 1/2, 2/5, 1/3]
        vae_stride = 16
        if 'vae_stride' in dynamic_scale_schedule:
            vae_stride = int(dynamic_scale_schedule.split('vae_stride')[-1])
        dynamic_resolution_h_w = {}
        for h_div_w in train_h_div_w_list:
            ratio = int(h_div_w*1000)/1000
            dynamic_resolution_h_w[ratio] = {}
            for pn in ['0.06M', '0.25M', '0.41M', '0.92M', '1M', '2M']:
                if pn == '0.06M':
                    scale = 8
                elif pn == '0.25M':
                    scale = 16
                elif pn == '0.41M':
                    scale = 20
                elif pn == '0.92M':
                    scale = 30
                elif pn == '1M':
                    scale = 32
                elif pn == '2M':
                    scale = 45
                if vae_stride == 16:
                    scale = scale * 2
                elif vae_stride == 32:
                    scale = scale * 1
                else:
                    raise ValueError(f'vae_stride {vae_stride} is not supported')
                area = scale * scale
                pw_float = math.sqrt(area / h_div_w)
                ph_float = pw_float * h_div_w
                ph, pw = int(np.round(ph_float)), int(np.round(pw_float))
                scales = [(ph,pw)]
                pixel = (scales[-1][0] * vae_stride, scales[-1][1] * vae_stride)
                dynamic_resolution_h_w[ratio][pn] = {
                    'pixel': pixel,
                    'scales': scales
                }
        for ratio in dynamic_resolution_h_w:
            for pn in dynamic_resolution_h_w[ratio]:
                base_scale_schedule = dynamic_resolution_h_w[ratio][pn]['scales']
                scales_in_one_clip = len(base_scale_schedule)
                dynamic_resolution_h_w[ratio][pn]['pt2scale_schedule'] = {}
                for pt in range(1, compressed_frames+1, 1):
                    dynamic_resolution_h_w[ratio][pn]['pt2scale_schedule'][pt] = [(pt, h, w) for h, w in base_scale_schedule]
                dynamic_resolution_h_w[ratio][pn]['image_scales'] = scales_in_one_clip
                dynamic_resolution_h_w[ratio][pn]['scales_in_one_clip'] = scales_in_one_clip
                dynamic_resolution_h_w[ratio][pn]['max_video_scales'] = len(dynamic_resolution_h_w[ratio][pn]['pt2scale_schedule'][compressed_frames])
                del dynamic_resolution_h_w[ratio][pn]['scales']
    else:
        raise ValueError(f'dynamic_scale_schedule={dynamic_scale_schedule} not implemented')
    return dynamic_resolution_h_w

def get_dynamic_resolution_meta(dynamic_scale_schedule, train_h_div_w_list, video_frames):
    dynamic_resolution_h_w = get_ratio2hws_pixels2scales(dynamic_scale_schedule, train_h_div_w_list, video_frames)
    h_div_w_templates = []
    for h_div_w in dynamic_resolution_h_w.keys():
        h_div_w_templates.append(h_div_w)
    h_div_w_templates = np.array(h_div_w_templates)
    return dynamic_resolution_h_w, h_div_w_templates

