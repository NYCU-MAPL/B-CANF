import torch


def uniform_noise(input, t=0.5):
    return torch.empty_like(input).uniform_(-t, t)


def quantize(input, mode="round", mean=None):
    if mode == "noise":
        return input + uniform_noise(input)
    else:
        if mean is not None:
            input = input - mean

        with torch.no_grad():
            diff = input.round() - input
        
        return input + diff


def scale_quant(input, scale=2**8):
    return quantize(input * scale) / scale


def noise_quant(input):
    return quantize(input, mode='noise')