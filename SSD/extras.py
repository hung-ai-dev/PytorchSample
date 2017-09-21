import torch.nn as nn
import torch.utils.model_zoo as model_zoo

extra_cfg = {
    '300': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256]
}

def add_extras(cfg, i, batch_norm = False):
    layers = []
    in_channels = i
    flag = False

    for k, v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S':
                layers.extend([nn.Conv2d(in_channels, cfg[k + 1], 
                            kernel_size = (1, 3)[flag], stride=2, padding=1)])
            else:
                layers.extend(nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag]))
            flag = not flag
        in_channels = v
    return layers