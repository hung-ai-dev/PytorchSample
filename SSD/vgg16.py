import torch.nn as nn
import torch.utils.model_zoo as model_zoo

cfg_vgg16 = {
    '300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512]
}

def vgg(cfg, i, batch_norm = False):
    layers = []
    in_channels = i
    for v in cfg:
        if v == 'M':
            layers.append(nn.MaxPool2d(2, 2))
        else:
            conv2d = nn.Conv2d(in_channels, v, 3, 1, 1)
            if batch_norm:
                layers.append(conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True))
            else:
                layers.append(conv2d, nn.ReLU(inplace=True))
        in_channels = v
    pool5 = nn.MaxPool2d(kernel_size = 3, stride = 1, padding = 1)
    conv6 = nn.Conv2d(512, 1024, kernel_size = 3,dilation = 6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size = 1)
    layers.extend([pool5, conv6, nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)])
    return layers