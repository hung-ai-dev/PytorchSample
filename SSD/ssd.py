import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from vgg16 import *
from extras import *
from multibox import *

import os


class SSD(nn.Module):
    """ Single Shot Multibox
    Base netork: VGG16
    Extra layers: multibox conv layers
    Each multibox layer branches into :
        1, conv2d for class conf scores
        2, conv2d for localization pred
        3, associated priorbox layer to produce default bounding boxes specific to the layer's feature map size

    Args:
        phase: test/train
        base: VGG16 layers
        extras: extra layers
        head: multibox consists of conf and loc layers
    """

    def __init__(self, phase, base, extras, loc, conf, num_classes):
        pass

    def forward(self, x):
        pass


def build_ssd(phase, size=300, num_classes=21):
    if phase != 'test' and phase != 'train':
        print('Error: Phase not recognized')
        return

    if size != 300:
        print('Error: Only support SSD300 currently!')
        return

    return SSD(phase,
               *(multibox(vgg(cfg_vgg16[str(size)], 3),
                          add_extras(extra_cfg[str(size)], 1024),
                          mbox_cfg[str(size)],
                          num_classes)),
               num_classes)
