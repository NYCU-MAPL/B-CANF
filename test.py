import os
import argparse
import csv
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm
from src.models.model import BCANF
from src.datasets.dataset import VideoDataset
from src.util.tools import seed_everything, estimate_bpp, get_setting, get_order
from src.util.metrics import psnr as psnr_fn
from src.util.metrics import MS_SSIM
from src.util.stream import Stream


class Tester:

    def __init__(self, args, model):
        self.args = args
        self.model = model
        self.psnr_fn = psnr_fn
        self.ssim_fn = MS_SSIM(data_range=1., reduction="none").to(args.device)

    @torch.no_grad()
    def quality_fn(self, rec, target):
        rec = rec.clamp(0, 1)
        psnr = self.psnr_fn(rec, target).mean()
        ssim = self.ssim_fn(rec, target).mean()
        return psnr, ssim

    @staticmethod
    @torch.no_grad()
    def rate_fn(likelihood, input):
        rate_y = estimate_bpp(likelihood['z'], input=input).mean()
        rate_z = estimate_bpp(likelihood['h'], input=input).mean()
        rate = rate_y + rate_z
        return rate
    
    @torch.no_grad()
    def test(self, eval_mode="test"):
        dataset = VideoDataset(self.args.src, self.args.num_frames, self.args.intra_period, self.args.gop_size)        
        test_loader = DataLoader(dataset, batch_size=1, num_workers=0, shuffle=False)
        order = get_order(dataset.pairs)
        
        _, img = next(test_loader.__iter__())
        shape = img.size()[-2:]
        num_frames = len(dataset)
        
        name = os.path.basename(self.args.src)
        os.makedirs(self.args.save_dir, exist_ok=True)
        
        if eval_mode == "compress":
            stream = Stream(os.path.join(self.args.save_dir, f"{name}.bin"), 'wb')
            stream.write_header(self.args.intra_period, self.args.gop_size, shape, num_frames)
        
        csv_report = open(os.path.join(self.args.save_dir, f"{name}.csv"), 'w', newline='')
        writer = csv.writer(csv_report, delimiter=',')
        writer.writerow(["frame_idx", "psnr", "ms-ssim", "rate", "coding mode"])
        
        frame_buffer = {}
        frame_bias = 1
        for i, (pair, x) in tqdm(enumerate(test_loader), total=len(test_loader)):
            img = x.to(self.args.device)
            pair = [int(p) for p in pair]
            
            inputs = {"xt": img}
            if len(pair) == 1:
                frame_idx = pair[0]
                mode = "i-frame"
                frame_type = -1
            elif len(pair) == 2:
                frame_idx = pair[1]
                inputs['x1'] = frame_buffer[pair[0]]
                mode = "b*-frame"
                frame_type = 2
            else:
                frame_idx = pair[1]
                inputs['x1'] = frame_buffer[pair[0]]
                inputs['x2'] = frame_buffer[pair[-1]]
                
                if order[pair[0]] > order[pair[-1]]:
                    inputs['x1'], inputs['x2'] = inputs['x2'], inputs['x1']
                
                mode = "b-frame"
                if abs(pair[-1] - pair[0]) > 2:
                    frame_type = 0
                else:
                    frame_type = 1
                
            output = self.model(inputs, mode, frame_type, shape, eval_mode)
            psnr, ssim = self.quality_fn(output["rec_frame"], img)
            
            if eval_mode == "test":
                rate = sum([self.rate_fn(l, img) for l in output["likelihood"]])
            elif eval_mode == "compress":
                bytes = sum([stream.writeStream(s['z']) + \
                             stream.writeStream(s['h']) for s in output["likelihood"]])
                rate = bytes * 8 / (shape[0] * shape[1])

            writer.writerow([frame_idx + frame_bias, float(psnr), float(ssim), float(rate), mode])
            csv_report.flush()

            frame_buffer[frame_idx] = output["rec_frame"]
            if i > 0 and i  % self.args.intra_period == 0:
                frame_buffer = {0: frame_buffer[self.args.intra_period]}
                frame_bias += self.args.intra_period

        if eval_mode == "compress":
            stream.close()

        df = pd.read_csv(csv_report.name)
        average = df.loc[:, df.columns != 'coding mode'].mean()
        writer.writerow(["average", float(average['psnr']), float(average['ms-ssim']), float(average['rate'])])
        csv_report.close()

    @torch.no_grad()
    def compress(self):
        self.test(eval_mode="compress")
        
    @torch.no_grad()
    def decompress(self):
        eval_mode = "decompress"
        stream = Stream(self.args.src, 'rb')
        self.args.intra_period, self.args.gop_size, shape, num_frames = stream.read_header()
        dataset = VideoDataset(self.args.src, num_frames, self.args.intra_period, 
                               self.args.gop_size, no_img=True)        
        test_loader = DataLoader(dataset, batch_size=1, num_workers=0, shuffle=False)
        order = get_order(dataset.pairs)
          
        name = "rec_" + os.path.basename(self.args.src)[:-4]
        self.args.save_dir = os.path.join(self.args.save_dir, name)
        os.makedirs(self.args.save_dir, exist_ok=True)
        
        frame_buffer = {}
        frame_bias = 1
        for i, pair in tqdm(enumerate(test_loader), total=len(test_loader)):
            pair = [int(p) for p in pair]
            
            inputs = {}
            if len(pair) == 1:
                frame_idx = pair[0]
                inputs['xt'] = stream.readStream()
                mode = "i-frame"
                frame_type = -1
            elif len(pair) == 2:
                frame_idx = pair[1]
                inputs['x1'] = frame_buffer[pair[0]]
                inputs['xt'] = {
                    "residual_stream": stream.readStream(),
                    "motion_stream": stream.readStream(),
                }
                mode = "b*-frame"
                frame_type = 2
            else:
                frame_idx = pair[1]
                inputs['x1'] = frame_buffer[pair[0]]
                inputs['x2'] = frame_buffer[pair[-1]]
                inputs['xt'] = {
                    "residual_stream": stream.readStream(),
                    "motion_stream": stream.readStream(),
                }
                
                if order[pair[0]] > order[pair[-1]]:
                    inputs['x1'], inputs['x2'] = inputs['x2'], inputs['x1']
                
                mode = "b-frame"
                if abs(pair[-1] - pair[0]) > 2:
                    frame_type = 0
                else:
                    frame_type = 1
                
            output = self.model(inputs, mode, frame_type, shape, eval_mode)
            filename = os.path.join(self.args.save_dir, f"frame_{frame_idx + frame_bias}.png")
            save_image(output["rec_frame"], filename)
            
            frame_buffer[frame_idx] = output["rec_frame"]
            if i > 0 and i  % self.args.intra_period == 0:
                frame_buffer = {0: frame_buffer[self.args.intra_period]}
                frame_bias += self.args.intra_period

        stream.close()


if __name__ == '__main__':
    seed_everything(888888)
    torch.backends.cudnn.deterministic = True
     
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--config",           type=str, default="./cfgs/bcanf.yaml")
    parser.add_argument('--src',              type=str, required=True)
    parser.add_argument('--ckpt',             type=str, required=True)
    parser.add_argument('--save_dir',         type=str, required=True)
    parser.add_argument('--mode',             type=str, choices=["test", "compress", "decompress"], required=True)
    parser.add_argument('--intra_period',     type=int, default=32, help="intra period")
    parser.add_argument('--gop_size',         type=int, default=16, help="gop size")
    parser.add_argument('--device',           type=str, choices=["cuda", "cpu"], default="cuda")
    parser.add_argument('--num_workers',      type=int, default=8)
    parser.add_argument('--num_frames',       type=int, default=-1, help="number of frame to test")
    args = parser.parse_args()  
    
    if args.mode in ["test", "compress"]:
        if args.num_frames < 0:
            parser.error("test/compress mode requires --num_frames")
    
    args = parser.parse_args()

    config = get_setting(args.config)
    model = BCANF(config).to(args.device)
    model.eval()
    model.load_state_dict(torch.load(args.ckpt, map_location=args.device), strict=True)
    
    tester = Tester(args, model)
    if args.mode == "test":
        tester.test()
    elif args.mode == "compress":
        tester.compress()
    elif args.mode == "decompress":
        tester.decompress()
    else:
        raise ValueError(f"invalid running mode: {args.mode}")
