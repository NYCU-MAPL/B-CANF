import torch
import torch.nn as nn
from src.modules.FS_modules import GridSynthNet
from src.modules.MP_modules import BiMotionPredict
from src.modules.ME_modules import SPyNet
from src.modules.codecs import get_coder_from_args
from src.layers.conditional_module import (conditional_warping,
                                           set_condition)
from src.util.alignment import Alignment


class ANFIC(nn.Module):

    def __init__(self, args) -> None:
        super().__init__()
        coder = get_coder_from_args(args)
        self.aligner = Alignment(divisor=64.)
        self.network = coder(in_channels=3)

    def forward(self, target):
        target = self.aligner.align(target)
        output = self.network(target)
        target_hat = self.aligner.resume(output['x_hat']) 
        return target_hat, output['likelihoods']
    
    def compress(self, target):
        target = self.aligner.align(target)
        output = self.network.compress(target)
        target_hat = self.aligner.resume(output['x_hat']) 
        return target_hat, output['streams']
    
    def decompress(self, streams, shape):
        new_shape = self.aligner.align_shape(shape)
        output = self.network.decompress(streams, new_shape)
        target_hat = self.aligner.resume(output['x_hat'])
        return target_hat


class MENet(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.aligner = Alignment(divisor=2**args.level)
        self.network = SPyNet(args.level)

    def forward(self, x1, xt):
        x1 = self.aligner.align(x1)
        xt = self.aligner.align(xt)
        esti_flow = self.network(x1, xt)
        esti_flow = self.aligner.resume(esti_flow)
        return esti_flow


class MPNet(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.aligner = Alignment(divisor=16.)
        self.network = BiMotionPredict(args.channels, args.num_blocks)

    def forward(self, x1, x2):
        x1 = self.aligner.align(x1)
        x2 = self.aligner.align(x2)
        pred_flow1, pred_flow2 = self.network(x1, x2)
        pred_flow1 = self.aligner.resume(pred_flow1)
        pred_flow2 = self.aligner.resume(pred_flow2)
        return pred_flow1, pred_flow2


class FrameSynNet(nn.Module):
    def __init__(self, args):
        super(FrameSynNet, self).__init__()
        self.aligner = Alignment(divisor=4.)
        self.network = GridSynthNet(args.channels, args.num_row, args.num_col)

    def forward(self, x1, x2, flow1, flow2):
        x1 = self.aligner.align(x1)
        x2 = self.aligner.align(x2)
        flow1 = self.aligner.align(flow1)
        flow2 = self.aligner.align(flow2)
        frame = self.network(x1, x2, flow1, flow2)
        frame = self.aligner.resume(frame)
        return frame


class MotionCodec(nn.Module):

    def __init__(self, args):
        super().__init__()
        coder = get_coder_from_args(args)
        self.aligner = Alignment(divisor=64.)
        self.network = coder(in_channels=4, cond_channels=4)
        
        conditional_warping(self.network, discrete=True, conditions=3, ver=2)
        self.frame_types = [
            torch.tensor([[0, 1, 0]], dtype=torch.float, requires_grad=False),
            torch.tensor([[1, 0, 0]], dtype=torch.float, requires_grad=False),
            torch.tensor([[0, 0, 1]], dtype=torch.float, requires_grad=False),
        ]

    def set_frame_type(self, level):
        fa = self.frame_types[level]
        set_condition(self.network, fa)

    def forward(self, esti_flow1, esti_flow2, pred_flow1, pred_flow2):
        target = torch.cat([esti_flow1, -esti_flow2], dim=1)
        pred = torch.cat([pred_flow1, -pred_flow2], dim=1)

        target = self.aligner.align(target)
        pred = self.aligner.align(pred)
        output = self.network(target, pred)
        
        target_hat = self.aligner.resume(output['x_hat'])
        rec_flow1 = target_hat[:, :2]
        rec_flow2 = -target_hat[:, 2:]

        return rec_flow1, rec_flow2, output['likelihoods']

    def compress(self, esti_flow1, esti_flow2, pred_flow1, pred_flow2):
        target = torch.cat([esti_flow1, -esti_flow2], dim=1)
        pred = torch.cat([pred_flow1, -pred_flow2], dim=1)

        target = self.aligner.align(target)
        pred = self.aligner.align(pred)
        output = self.network.compress(target, pred)
        
        target_hat = self.aligner.resume(output['x_hat'])
        rec_flow1 = target_hat[:, :2]
        rec_flow2 = -target_hat[:, 2:]

        return rec_flow1, rec_flow2, output['streams']
    
    def decompress(self, streams, pred_flow1, pred_flow2, shape):
        pred = torch.cat([pred_flow1, -pred_flow2], dim=1)
        pred = self.aligner.align(pred)
        output = self.network.decompress(streams, pred, pred.size()[-2:])
        target_hat = self.aligner.resume(output['x_hat'])
        rec_flow1 = target_hat[:, :2]
        rec_flow2 = -target_hat[:, 2:]
        return rec_flow1, rec_flow2


class ResidualCodec(nn.Module):

    def __init__(self, args):
        super().__init__()
        coder = get_coder_from_args(args)
        self.aligner = Alignment(divisor=64.)
        self.network = coder(in_channels=3, cond_channels=3)
            
        conditional_warping(self.network, discrete=True, conditions=3, ver=2)
        self.frame_types = [
            torch.tensor([[0, 1, 0]], dtype=torch.float, requires_grad=False),
            torch.tensor([[1, 0, 0]], dtype=torch.float, requires_grad=False),
            torch.tensor([[0, 0, 1]], dtype=torch.float, requires_grad=False),
        ]

    def set_frame_type(self, level):
        fa = self.frame_types[level]
        set_condition(self.network, fa)

    def forward(self, target, pred):
        target = self.aligner.align(target)
        pred = self.aligner.align(pred)
        output = self.network(target, pred)

        target_hat = self.aligner.resume(output['x_hat'])
        return target_hat, output['likelihoods']
    
    def compress(self, target, pred):
        target = self.aligner.align(target)
        pred = self.aligner.align(pred)
        output = self.network.compress(target, pred)

        target_hat = self.aligner.resume(output['x_hat'])
        return target_hat, output['streams']
    
    def decompress(self, streams, pred, shape):
        pred = self.aligner.align(pred)
        output = self.network.decompress(streams, pred, pred.size()[-2:])
        target_hat = self.aligner.resume(output['x_hat'])
        return target_hat


MODULES = {
    "ANFIC": ANFIC,
    "MENet": MENet,
    "MPNet": MPNet,
    "FrameSynNet": FrameSynNet,
    "MotionCodec": MotionCodec,
    "ResidualCodec": ResidualCodec,
}