from torch.utils.data import Dataset as torchData
from torchvision import transforms
from torchvision.datasets.folder import default_loader as imgloader

to_tensor = transforms.ToTensor()


def get_next_frame(pairs, intra_period, gop_size, idx_l, idx_r, mode):
    if idx_r > intra_period:
        return
    
    if mode == "i-frame":
        idx = idx_l
        pairs.append((idx,))
        
        if idx == 0:
            if (idx+gop_size) % intra_period == 0:
                get_next_frame(pairs, intra_period, gop_size, idx+gop_size, idx+gop_size, "i-frame")
            else:
                get_next_frame(pairs, intra_period, gop_size, idx, idx+gop_size, "p-frame")
        else:
            get_next_frame(pairs, intra_period, gop_size, idx-gop_size, idx, "b-frame")
    elif mode == "p-frame":
        idx = idx_r
        pairs.append((idx_l, idx))
        get_next_frame(pairs, intra_period, gop_size, idx_l, idx, "b-frame")
        
        if (idx+gop_size) % intra_period == 0:
            get_next_frame(pairs, intra_period, gop_size, idx+gop_size, idx+gop_size, "i-frame")
        else:
            get_next_frame(pairs, intra_period, gop_size, idx, idx+gop_size, "p-frame")
    else:
        if idx_l != idx_r - 1:
            idx = (idx_l+idx_r)//2
            pairs.append((idx_l, idx, idx_r))
            get_next_frame(pairs, intra_period, gop_size, idx_l, idx, "b-frame")
            get_next_frame(pairs, intra_period, gop_size, idx, idx_r, "b-frame")


def get_coding_pairs(intra_period, gop_size):        
    pairs = []
    get_next_frame(pairs, intra_period, gop_size, 0, 0, "i-frame")
    
    
    assert len(pairs) == intra_period + 1, \
        f"Number of pairs: {len(pairs)} is greater than {intra_period} + 1: {pairs}"

    target = [p[0] if len(p) == 1 else p[1] for p in pairs]
    for i in range(intra_period + 1):
        assert i in target, f"Frame: {i} does not covered by {pairs}"

    if gop_size == 1:
        pairs.pop(-1)

    return pairs


class VideoDataset(torchData):

    def __init__(self, root, num_frame, intra_period, gop_size, no_img=False):
        super(VideoDataset, self).__init__()
        self.root = root
        self.num_frame = (num_frame // intra_period) * intra_period + 1
        self.intra_period = intra_period
        self.gop_size = gop_size
        self.pairs = get_coding_pairs(intra_period, gop_size)
        self.no_img = no_img

    def __len__(self):
        return self.num_frame

    def __getitem__(self, idx):
        if idx == 0:
            intra_idx = 0
            pair_idx = 0
        else:
            intra_idx = ((idx - 1) // self.intra_period) * self.intra_period
            pair_idx = (idx - 1) % self.intra_period + 1

        pair = self.pairs[pair_idx]
        frame_idx = pair[1 if len(pair) > 1 else 0] + intra_idx + 1

        if not self.no_img:
            raw_path = f"{self.root}/frame_{frame_idx}.png"
            img = to_tensor(imgloader(raw_path))
            return pair, img
        else:
            return pair