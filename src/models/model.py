import torch
from torch import nn
from src.modules.modules import MODULES
from types import SimpleNamespace


class BCANF(nn.Module):

    def __init__(self, config, **kwargs):
        super().__init__()
        self.module_table = config["MODULES"]
        for name, mprop in self.module_table.items():
            args = SimpleNamespace(**mprop)
            self.__setattr__(name, MODULES[mprop["TYPE"]](args))

    def forward_intra_frame(self, x, shape, eval_mode):
        if eval_mode == "test":
            rec_frame, likelihood = self.IntraCodec(x)
        elif eval_mode == "compress":
            rec_frame, stream = self.IntraCodec.compress(x)
            likelihood = stream
        else:
            rec_frame = self.IntraCodec.decompress(x, shape)
            likelihood = None
            
        return rec_frame, [likelihood]

    def forward_inter_frame(self, x1, x2, xt, mode, frame_type, shape, eval_mode):
        self.MotionCodec.set_frame_type(frame_type)
        self.ResidualCodec.set_frame_type(frame_type)

        if eval_mode != "decompress":
            esti_flow1 = self.MENet(x1, xt)
            if mode == "b-frame":
                esti_flow2 = self.MENet(x2, xt)
            else:
                esti_flow2 = -esti_flow1

        if mode == "b-frame":
            pred_flow1, pred_flow2 = self.MPNet(x1, x2)
        else:
            pred_flow1 = pred_flow2  = torch.zeros_like(x1)[:, :2]

        if eval_mode == "test":
            rec_flow1, rec_flow2, likelihood_m = self.MotionCodec(esti_flow1, esti_flow2, 
                                                                  pred_flow1, pred_flow2)
        elif eval_mode == "compress":
            rec_flow1, rec_flow2, stream_m = self.MotionCodec.compress(esti_flow1, esti_flow2, 
                                                                       pred_flow1, pred_flow2)
            likelihood_m = stream_m
        else:
            rec_flow1, rec_flow2 = self.MotionCodec.decompress(xt['motion_stream'],
                                                               pred_flow1, pred_flow2, shape)
            likelihood_m = None

        if mode == "b*-frame":
            rec_flow2 = -rec_flow2

        cond_frame = self.FSNet(x1, x2, rec_flow1, rec_flow2).clamp(0., 1.)
        
        if eval_mode == "test":
            rec_frame, likelihood_r = self.ResidualCodec(xt, cond_frame)
        elif eval_mode == "compress":
            rec_frame, stream_r = self.ResidualCodec.compress(xt, cond_frame)
            likelihood_r = stream_r
        else:
            rec_frame = self.ResidualCodec.decompress(xt['residual_stream'], cond_frame, shape)
            likelihood_r = None
        
        return rec_frame, [likelihood_r, likelihood_m]

    def forward(self, inputs, mode, frame_type, shape, eval_mode):
        if mode == "i-frame":
            rec_frame, likelihood = self.forward_intra_frame(inputs["xt"], shape, eval_mode)
        elif mode == "b-frame":
            rec_frame, likelihood = self.forward_inter_frame(inputs["x1"], inputs["x2"], inputs["xt"], 
                                                             "b-frame", frame_type, shape, eval_mode)
        elif mode == "b*-frame":
            rec_frame, likelihood = self.forward_inter_frame(inputs["x1"], inputs["x1"], inputs["xt"], 
                                                             "b*-frame", frame_type, shape, eval_mode)
        else:
            raise ValueError(f"invalid coding mode: {mode}")

        return {
            "rec_frame": rec_frame,
            "likelihood": likelihood
        }