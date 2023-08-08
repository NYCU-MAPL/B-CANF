import os
import yaml
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.util.math import lower_bound


def seed_everything(seed):
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_setting(config_path):
    with open(config_path, 'r') as f:
        setting = yaml.full_load(f)
         
    return setting


def estimate_bpp(likelihood, input=None, eps=1e-9):

    assert torch.is_tensor(input) and input.dim() > 2
    num_pixels = np.prod(input.size()[-2:])

    if torch.is_tensor(likelihood):
        likelihood = [likelihood]

    lll = 0
    for ll in likelihood:
        lll = lll + lower_bound(ll, eps).log().flatten(1).sum(1)

    return lll / (-np.log(2.) * num_pixels)


def resize(x, size):
    return F.interpolate(x, size, mode='bilinear', align_corners=False) 


def flow_interpolate(flow, size):
    h, w = flow.size()[-2:]
    new_h, new_w = size
    scale_h = new_h / h
    scale_w = new_w / w
    scale = torch.tensor([scale_w, scale_h], dtype=flow.dtype, device=flow.device).view(1, 2, 1, 1)
    return resize(flow, size) * scale


def get_order(pairs):
    num_frame = max([max(p) for p in pairs]) + 1
    order = [0 for _ in range(num_frame)]

    for p in pairs:
        if len(p) == 1:
            order[p[0]] = 0
        elif len(p) == 2:
            order[p[1]] = order[p[0]] + 1
        elif len(p) == 3:
            order[p[1]] = min(order[p[0]], order[p[2]]) + 1

    return order


def Conv2d(in_channels, out_channels, kernel_size=5, stride=1, *args, **kwargs):
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                     padding=(kernel_size - 1) // 2, *args, **kwargs)


def ConvTranspose2d(in_channels, out_channels, kernel_size=5, stride=1, *args, **kwargs):
    return nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride,
                              padding=(kernel_size - 1) // 2, output_padding=stride-1, *args, **kwargs)