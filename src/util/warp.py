import torch
Grids = {}

def torch_warp(input, flow):
    if str(flow.size()) not in Grids:
        B, _, H, W = flow.size()
        gridX = torch.linspace(-1.0, 1.0, W).view(1, 1, 1, -1).expand(B, -1, H, -1)
        gridY = torch.linspace(-1.0, 1.0, H).view(1, 1, -1, 1).expand(B, -1, -1, W)
        Grids[str(flow.size())] = torch.cat([gridX, gridY], 1).cuda()

    H, W = input.size()[-2:]
    dflow = torch.cat([flow[:, [0], :, :] / ((W - 1.0) / 2.0), 
                       flow[:, [1], :, :] / ((H - 1.0) / 2.0)], 1)
    grid = (Grids[str(flow.size())] + dflow).permute(0, 2, 3, 1)
    return torch.nn.functional.grid_sample(input=input, grid=grid, mode='bilinear',
                                           padding_mode='border', align_corners=True)