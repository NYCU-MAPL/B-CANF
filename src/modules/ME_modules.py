import torch
import torch.nn as nn
import torch.nn.functional as F
from src.util.warp import torch_warp as warp
from src.util.tools import flow_interpolate


class SPyBlock(nn.Sequential):

    def __init__(self):
        super().__init__(
            nn.Conv2d(8, 32, kernel_size=7, padding=3),
            nn.ReLU(inplace=False),
            nn.Conv2d(32, 64, kernel_size=7, padding=3),
            nn.ReLU(inplace=False),
            nn.Conv2d(64, 32, kernel_size=7, padding=3),
            nn.ReLU(inplace=False),
            nn.Conv2d(32, 16, kernel_size=7, padding=3),
            nn.ReLU(inplace=False),
            nn.Conv2d(16, 2, kernel_size=7, padding=3)
        )

    def forward(self, flow_course, im1, im2):
        flow = flow_interpolate(flow_course, im2.size()[2:])
        res = super().forward(torch.cat([im1, warp(im2, flow), flow], dim=1))
        flow_fine = res + flow
        return flow_fine


class SPyNet(nn.Module):

    def __init__(self, level=5):
        super(SPyNet, self).__init__()
        self.level = level
        self.Blocks = nn.ModuleList([SPyBlock() for _ in range(level+1)])
        self.register_buffer('mean', torch.Tensor(
            [.406, .456, .485]).view(-1, 1, 1))
        self.register_buffer('std', torch.Tensor(
            [.225, .224, .229]).view(-1, 1, 1))

    def norm(self, input):
        return input.sub(self.mean).div(self.std)

    def forward(self, im2, im1):
        volume = [torch.cat([self.norm(im1), self.norm(im2)], dim=1)]
        for _ in range(self.level):
            volume.append(F.avg_pool2d(volume[-1], kernel_size=2))

        flows = [torch.zeros_like(volume[-1][:, :2])]
        for l, layer in enumerate(self.Blocks):
            flow = layer(flows[-1], *volume[self.level-l].chunk(2, 1))
            flows.append(flow)

        return flows[-1]
