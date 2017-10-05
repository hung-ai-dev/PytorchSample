import os.path as osp

import fcn
import torch.nn as nn


class Segnet(nn.Module):
    """
        input: 256x256
        output: cx256x256
    """

    def __init__(self, n_class):
        def conv_bn(in_c, out_c, stride=1):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, stride, 1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True)
            )

        def conv_bn_dw(in_c, out_c, stride=1):
            return nn.Sequential(
                nn.Conv2d(in_c, in_c, 3, stride, 1, groups=in_c, bias=False),
                nn.BatchNorm2d(in_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_c, out_c, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True)
            )

        def deconv_bn(in_c, out_c, stride=1):
            return nn.Sequential(
                nn.ConvTranspose2d(in_c, out_c, 3, stride, 1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True)
            )

        def deconv_bn_dw(in_c, out_c, stride=1):
            return nn.Sequential(
                nn.ConvTranspose2d(in_c, out_c, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(out_c, out_c, 3, stride,
                                   1, groups=out_c, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True)
            )

        super().__init__()
        r = []
        self.decoder = []

        self.encode = nn.Sequential(
            conv_bn(3, 32, 2),
            conv_bn_dw(32, 64, 1),
            conv_bn_dw(64, 128, 2),
            conv_bn_dw(128, 128, 1),
            conv_bn_dw(128, 256, 2),
            conv_bn_dw(256, 256, 1),
            conv_bn_dw(256, 512, 2),
            conv_bn_dw(512, 512, 1),
            conv_bn_dw(512, 512, 1),
        )
       

        self.decode9 = deconv_bn_dw(512, 512, 1),
        self.decode8 = deconv_bn_dw(512, 512, 1),
        self.decode7 = deconv_bn_dw(512, 256, 2),
        self.decode6 = deconv_bn_dw(256, 256, 1),
        self.decode5 = deconv_bn_dw(256, 128, 2),
        self.decode4 = deconv_bn_dw(128, 128, 1),
        self.decode3 = deconv_bn_dw(128, 64, 2),
        self.decode2 = deconv_bn_dw(64, 32, 1),
        self.decode1 = deconv_bn(32, n_class, 2)
        # self.decoder.extend([decode1, decode2, decode3, decode4, decode5, decode6, decode7, decode8, decode9])

    def forward(self, x):
        x = self.encode(x)

        h_in, w_in = s.size()
        h_out, w_out = h_in * 2, w_in * 2

        x = self.decode9(x, (h_in, w_in))
        x = self.decode8(x, (h_in, w_in))
        h_out, w_out = h_in * 2, w_in * 2
        x = self.decode7(x, (h_out, w_out))
        x = self.decode6(x, (h_out, w_out))
        h_out, w_out = h_out * 2, w_out * 2
        x = self.decode5(x, (h_out, w_out))
        x = self.decode4(x, (h_out, w_out))
        h_out, w_out = h_out * 2, w_out * 2
        x = self.decode3(x, (h_out, w_out))
        x = self.decode2(x, (h_out, w_out))
        h_out, w_out = h_out * 2, w_out * 2
        x = self.decode1(x, (h_out, w_out))
