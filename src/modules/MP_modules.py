import torch
import torch.nn as nn
from src.util.warp import torch_warp as warp
from src.util.tools import Conv2d, resize


def ConvBlock(in_planes, out_planes, kernel_size=3, stride=1):
    return nn.Sequential(
        Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride),
        nn.PReLU(out_planes)
    )
    

class ResidualBlock(nn.Sequential):
    
    def __init__(self, chs, num_blocks):
        super().__init__(
            *[ConvBlock(chs, chs) for _ in range(num_blocks)]
        )
        
    def forward(self, input):
        return super().forward(input) + input
        

class MPBlock(nn.Module):
    def __init__(self, in_chs, h_chs, num_blocks):
        super(MPBlock, self).__init__()
        self.layers = nn.Sequential(
            ConvBlock(in_chs, h_chs//2, 3, 2),
            ConvBlock(h_chs//2, h_chs, 3, 2),
            ResidualBlock(h_chs, num_blocks),
            nn.ConvTranspose2d(h_chs, 4, 4, 2, 1) 
        )

    def forward(self, x, flow, scale):
        ori_size = x.size()[-2:]
        new_size = [s // scale for s in ori_size]
        
        if scale != 1:
            x = resize(x, new_size)
        
        if flow != None:
            flow = resize(flow, new_size) / scale
            x = torch.cat((x, flow), 1)

        flow = self.layers(x)
        flow = resize(flow, ori_size) * (2 * scale)
        return flow


class BiMotionPredict(nn.Module):

    def __init__(self, channels, num_blocks):
        super(BiMotionPredict, self).__init__()
        self.blocks = nn.ModuleList([
            MPBlock(6, channels[0], num_blocks[0]),
            *[MPBlock(12+4, c, n) for c, n in zip(channels[1:], num_blocks[1:])]
        ])

    def forward(self, x0, x1, scale=[4,2,1]):
        img0 = x0
        img1 = x1

        for i, block in enumerate(self.blocks):
            if i > 0:
                inputs = torch.cat((img0, img1, warped_img0, warped_img1), 1)
                flow_d = block(inputs, flow, scale=scale[i])
                flow = flow + flow_d
            else:
                flow = block(torch.cat((img0, img1), 1), None, scale=scale[i])

            warped_img0 = warp(img0, flow[:, :2])
            warped_img1 = warp(img1, flow[:, 2:4])
      
        return flow.chunk(2, 1) 