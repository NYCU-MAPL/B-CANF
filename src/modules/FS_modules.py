from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.util.warp import torch_warp as warp
from src.util.tools import Conv2d


def CovnBlock(in_channels, out_channels, stride=1):
	return torch.nn.Sequential(
		nn.PReLU(),
		Conv2d(in_channels, out_channels, kernel_size=3, stride=stride),
        nn.PReLU(),
		Conv2d(out_channels, out_channels, kernel_size=3, stride=stride)
	)


def DownsampleBlock(in_channels, out_channels, stride=2):
	return torch.nn.Sequential(
		nn.PReLU(),
		Conv2d(in_channels, out_channels, kernel_size=3, stride=stride),
        nn.PReLU(),
		Conv2d(out_channels, out_channels, kernel_size=3, stride=1)
	)


def UpsampleBlock(in_channels, out_channels, stride=1):
	return torch.nn.Sequential(
		nn.Upsample(scale_factor=2, mode='bilinear'),
		nn.PReLU(),
		Conv2d(in_channels, out_channels, kernel_size=3, stride=stride),
        nn.PReLU(),
		Conv2d(out_channels, out_channels, kernel_size=3, stride=stride),
	)


def BackboneBlock(in_channels, out_channels, stride=2):
	return torch.nn.Sequential(
		Conv2d(in_channels, out_channels, kernel_size=3, stride=stride),
        nn.PReLU(),
		Conv2d(out_channels, out_channels, kernel_size=3, stride=1),
        nn.PReLU()
	)


class ColumnBlock(nn.Module):
    def __init__(self, channels: List, down: bool) -> None:
        super(ColumnBlock, self).__init__()
        self.down = down
        
        if down:
            bridge = DownsampleBlock 
        else:
            bridge = UpsampleBlock
            channels = channels[::-1]
        
        self.resblocks = nn.ModuleList([CovnBlock(c, c, stride=1) for c in channels])
        self.bridge = nn.ModuleList([bridge(cin, cout) for cin, cout in zip(channels[:-1], channels[1:])])

    def forward(self, inputs) -> List:
        outputs = []

        if not self.down:
            inputs = inputs[::-1]

        for i, x in enumerate(inputs):
            out = self.resblocks[i](x)

            if i > 0:
                out += self.bridge[i-1](outputs[-1])

            outputs.append(out)

        if not self.down:
            outputs = outputs[::-1]

        return outputs


class Backbone(nn.Module):
    def __init__(self, hidden_channels: List) -> None:
        super().__init__()
        self.backbone = nn.ModuleList([BackboneBlock(cin, cout, stride=1 if cin == 3 else 2)
                                       for cin, cout in zip(hidden_channels[:-1], hidden_channels[1:])])

    def forward(self, x):
        feats = []
        for m in self.backbone:
            feats.append(m(x))
            x = feats[-1]

        return feats


class GridNet(nn.Module):
    def __init__(self, in_channels: List, hidden_channels: List, columns, out_channels: int):
        super(GridNet, self).__init__()
        self.heads = nn.ModuleList([CovnBlock(i, c, stride=1)
                                    for i, c in zip(in_channels, [hidden_channels[0]] + hidden_channels)])
        
        self.downs = nn.ModuleList([nn.Identity()])
        self.downs.extend([DownsampleBlock(cin, cout) 
                           for cin, cout in zip(hidden_channels[:-1], hidden_channels[1:])])
    
        columns -= 1 # minus 1 for heads
        self.columns = nn.Sequential(*[ColumnBlock(hidden_channels, n < columns//2) for n in range(columns)])
        self.tail = CovnBlock(hidden_channels[0], out_channels, stride=1)


    def forward(self, inputs):        
        feats = []
        for i, x in enumerate(inputs):
            feat = self.heads[i](x)
           
            if i > 0:
                feat += self.downs[i-1](feats[-1]) 
            
            feats.append(feat)

        feats.pop(0)
        feats = self.columns(feats)
        output = self.tail(feats[0])

        return output, feats
    

class GridSynthNet(nn.Module):

    def __init__(self, channels, num_row, num_col):
        super(GridSynthNet, self).__init__()
        self.backbone = Backbone(channels)
        cc = [c * 2 for c in channels]
        self.synth = GridNet(cc, channels[1:], num_col, num_row)

    def forward(self, x0, x1, flow0, flow1):
        feats0 = self.backbone(x0)
        feats1 = self.backbone(x1)
        warped_img0 = warp(x0, flow0)
        warped_img1 = warp(x1, flow1)
        flow = torch.cat([flow0, flow1], dim=1)

        warped_feats = [torch.cat([warped_img0, warped_img1], dim=1)]
        for level, (f0, f1) in enumerate(zip(feats0, feats1)):
            s = 2**level
            flow_scaled = F.interpolate(flow, scale_factor = 1. / s, mode="bilinear", align_corners=False) * 1. / s
            warped_f0 = warp(f0, flow_scaled[:, :2])
            warped_f1 = warp(f1, flow_scaled[:, 2:4])
            warped_feats.append(torch.cat([warped_f0, warped_f1], dim=1))

        frame, _ = self.synth(warped_feats)
        return frame
