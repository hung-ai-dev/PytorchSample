import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

class Flatten(nn.Module):
    def forward(self, x):
        N, C, H, W = x.size()
        return x.view(N, -1)

class MobileNet(nn.Module):
    def __init__(self):
        super().__init__()

        def conv_bn(in_channel, out_channel, stride = 1):
            return nn.Sequential(
                nn.Conv2d(in_channel, out_channel, 3, stride, 1, bias=False),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(inplace=True)
            )

        def conv_bn_dw(in_channel, out_channel, stride = 1):
            return nn.Sequential(
                nn.Conv2d(in_channel, in_channel, 3, stride, 1, groups=in_channel, bias=False),
                nn.BatchNorm2d(in_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channel, out_channel, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(inplace=True)
            )
        
        self.features = nn.Sequential(
            # conv_bn(3,  32, 2), 
            # conv_bn_dw( 32,  64, 1),
            # conv_bn_dw( 64, 128, 2),
            # conv_bn_dw(128, 128, 1),
            conv_bn(1, 8, 1),       # 1x28x28 -> 8x28x28
            conv_bn_dw(8, 8, 1),    # 8x28x28 -> 8x28x28
            conv_bn_dw(8, 16, 2),   # 8x28x28 -> 16x14x14
            conv_bn_dw(16, 16, 1),  # 16x14x14 -> 16x14x14
            conv_bn_dw(16, 16, 1),  # 16x14x14 -> 16x14x14
            conv_bn_dw(16, 32, 2),  # 16x14x14 -> 32x7x7
            conv_bn_dw(32, 32, 1),  # 32x7x7 -> 32x7x7
            conv_bn_dw(32, 32, 1),  # 32x7x7 -> 32x7x7
            nn.AvgPool2d(7),        # 32x7x7 -> 32x1x1
            Flatten(),              
        )

        self.fc = nn.Linear(32, 10)

    def forward(self, x):
        return F.log_softmax(self.fc(self.features(x)))