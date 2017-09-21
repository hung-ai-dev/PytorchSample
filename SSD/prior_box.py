import torch
import math
from itertools import product

class PriorBox(object):
    def __init__(self, cfg):
        super().__init__()
        self.image_size = cfg['min_dim']
        self.num_priors = len(cfg['aspect_ratios'])
        self.variance = cfg['variance'] or [0.1]
        self.feature_maps = cfg['feature_maps']
        self.min_sizes = cfg['min_sizes']
        self.max_sizes = cfg['max_sizes']
        self.steps = cfg['steps']
        self.aspect_ratios = cfg['aspect_ratios']
        self.clip = cfg['clip']
        self.version = cfg['name']
        for v in self.variance:
            if v <= 0:
                raise ValueError('Variances must be greater than 0')

    def forward(self):
        box = []
        if self.version == 'v2':
            for k, f in enumerate(self.feature_maps):
                for i, j in product(range(f), repeat=2):
                    f_k = self.image_size / self.steps[k]
                    cx = (j + 0.5) / f_k
                    cy = (i + 0.5) / f_k

                    # aspect_ratio: 1
                    # rel size: min_size
                    s_k = self.min_sizes[k]/self.image_size
                    box.append([cx, cy, s_k, s_k])

                    # aspect_ratio: 1
                    # rel size: sqrt(s_k * s_(k+1))
                    s_k_prime = math.sqrt(s_k * (self.max_sizes[k]/self.image_size))
                    box.append([cx, cy, s_k_prime, s_k_prime])

                    # rest of aspect ratios
                    for ar in self.aspect_ratios[k]:
                        box.append([cx, cy, s_k * math.sqrt(ar), s_k / math.sqrt(ar)])
                        box.append([cx, cy, s_k / math.sqrt(ar), s_k * math.sqrt(ar)])
        output = torch.Tensor(box).view(-1, 4)
        if self.clip:
            output.clamp_(max = 1, min = 0)
        return output