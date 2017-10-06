import os.path as osp

import fcn
import torch.nn as nn


class ConvBn(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class ConvBnDw(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.block == nn.Sequential(
            nn.Conv2d(in_c, in_c, 3, stride, 1, groups=in_c, bias=False),
            nn.BatchNorm2d(in_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_c, out_c, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x)
        return self.block(x)


class DeconvBn(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_c, out_c, 3, stride, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, size):
        x = self.conv(x, size)
        x = self.bn(x)
        return self.relu(x)


class DeconvBnDw(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.stride = stride
        self.conv1 = nn.ConvTranspose2d(in_c, out_c, 1, 1, 0, bias=False),
        self.bn1 = nn.BatchNorm2d(out_c),
        self.relu1 = nn.ReLU(inplace=True),

        self.conv2 = nn.ConvTranspose2d(out_c, out_c, 3, stride, 1, groups=out_c, bias=False),
        self.bn2 = nn.BatchNorm2d(out_c),
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x, size):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x, size)
        x = self.bn2(x)
        x = self.relu2(x)

class Segnet(nn.Module):
    """
        input: 256x256
        output: cx256x256
    """

    def __init__(self, n_class):
        super().__init__()
        self.encode = nn.Sequential(
            ConvBn(3, 32, 2),
            ConvBnDw(32, 64, 1),
            ConvBnDw(64, 128, 2),
            ConvBnDw(128, 128, 1),
            ConvBnDw(128, 256, 2),
            ConvBnDw(256, 256, 1),
            ConvBnDw(256, 512, 2),
            ConvBnDw(512, 512, 1),
            ConvBnDw(512, 512, 1),
        )

        self.decode4 = DeconvBnDw(512, 512, 1)
    def forward(self, x):
        pass