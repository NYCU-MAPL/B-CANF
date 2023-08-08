import torch
import torch.nn.functional as F
from torch import nn


class ConditionalLayer(nn.Module):

    def __init__(self, module: nn.Module, discrete=False, conditions: int = 1, ver=2, h_channels=16, **kwargs):
        super(ConditionalLayer, self).__init__()
        self.m = module
        self.ver = ver
        self.discrete = discrete
        self.out_channels = module.out_channels

        if self.ver == 1:
            self.weight = nn.Parameter(torch.Tensor(conditions, self.out_channels*2))
            nn.init.kaiming_normal_(self.weight)
        elif self.ver == 2:
            self.affine = nn.Sequential(
                nn.Linear(conditions, h_channels),
                nn.Sigmoid(),
                nn.Linear(h_channels, self.out_channels * 2, bias=False)
            )
        else:
            raise ValueError(f"Do not support ver = {ver}")

    def _set_condition(self, condition):
        self.condition = condition

    def forward(self, *input, condition=None, **kwargs):
        output = self.m(*input)

        if condition is None:
            condition = self.condition

        if isinstance(condition, tuple):
            scale, bias = condition
        else:
            if condition.device != output.device:
                condition = condition.to(output.device)
            
            if self.ver == 1:
                condition = condition.mm(self.weight)
            else:
                condition = self.affine(condition)

            scale, bias = condition.view(
                condition.size(0), -1, *(1,)*(output.dim()-2)).chunk(2, dim=1)

            self.condition = (scale, bias)
            
        output = output * F.softplus(scale) + bias
        return output.contiguous()


def conditional_warping(m: nn.Module, types=(nn.modules.conv._ConvNd), **kwargs):
    def dfs(sub_m: nn.Module, prefix=""):
        for n, chd_m in sub_m.named_children():
            if dfs(chd_m, prefix+"."+n if prefix else n):
                setattr(sub_m, n, ConditionalLayer(chd_m, **kwargs))
        else:
            if isinstance(sub_m, types) and sub_m.in_channels > 10 and sub_m.out_channels > 10:
                return True
            else:
                pass
        return False

    dfs(m)


def set_condition(model, condition):
    for m in model.modules():
        if isinstance(m, ConditionalLayer):
            m._set_condition(condition)