class Stream:
    
    def __init__(self, path: str, mode):
        self.bin = open(path, mode)
        
    def write_header(self, intra_period, gop_size, shape, num_frames):
        self.bin.write(intra_period.to_bytes(1, byteorder='big', signed=False))
        self.bin.write(gop_size.to_bytes(1, byteorder='big', signed=False))
        self.bin.write(shape[0].to_bytes(2, byteorder='big', signed=False))
        self.bin.write(shape[1].to_bytes(2, byteorder='big', signed=False))
        self.bin.write(num_frames.to_bytes(4, byteorder='big', signed=False))
        
    def read_header(self):
        intra_period = int.from_bytes(self.bin.read(1), byteorder='big', signed=False)
        gop_size = int.from_bytes(self.bin.read(1), byteorder='big', signed=False)
        shape = [None, None]
        shape[0] = int.from_bytes(self.bin.read(2), byteorder='big', signed=False)
        shape[1] = int.from_bytes(self.bin.read(2), byteorder='big', signed=False)
        num_frames = int.from_bytes(self.bin.read(4), byteorder='big', signed=False)
        return intra_period, gop_size, shape, num_frames

    def writeStream(self, strings):
        total = 0
        for i, s in enumerate(strings):
            if isinstance(s, int):
                num_bytes = int(4).to_bytes(4, byteorder='big', signed=False)
                s = s.to_bytes(4, byteorder='big', signed=False)
            else:
                num_bytes = len(s).to_bytes(4, byteorder='big', signed=False)
            self.bin.write(num_bytes)
            self.bin.write(s)
            total += 4 + len(s)
        return total

    def readStream(self):
        strings = []
        num = 4
        for _ in range(num):
            num_bytes = int.from_bytes(self.bin.read(4), byteorder='big', signed=False)
            strings.append(self.bin.read(num_bytes))
        return [[strings[0], strings[1]], [strings[2], strings[3]]]

    def flush(self):
        self.bin.flush()

    def close(self):
        self.bin.close()