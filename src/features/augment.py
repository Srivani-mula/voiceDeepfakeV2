import torch
import random


class SpecAugment:
    def __init__(
        self,
        time_mask_param=30,
        freq_mask_param=15,
        num_time_masks=2,
        num_freq_masks=2,
    ):
        self.time_mask_param = time_mask_param
        self.freq_mask_param = freq_mask_param
        self.num_time_masks = num_time_masks
        self.num_freq_masks = num_freq_masks

    def time_mask(self, spec):
        _, _, T = spec.shape
        for _ in range(self.num_time_masks):
            t = random.randint(0, self.time_mask_param)
            t0 = random.randint(0, max(1, T - t))
            spec[:, :, t0 : t0 + t] = 0
        return spec

    def freq_mask(self, spec):
        _, F, _ = spec.shape
        for _ in range(self.num_freq_masks):
            f = random.randint(0, self.freq_mask_param)
            f0 = random.randint(0, max(1, F - f))
            spec[:, f0 : f0 + f, :] = 0
        return spec

    def __call__(self, spec):
        spec = self.time_mask(spec)
        spec = self.freq_mask(spec)
        return spec


class RandomGain:
    def __init__(self, min_gain=0.8, max_gain=1.2):
        self.min_gain = min_gain
        self.max_gain = max_gain

    def __call__(self, spec):
        gain = random.uniform(self.min_gain, self.max_gain)
        return spec * gain
