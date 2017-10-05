import argparse
import datetime
import os
import os.path as osp
import shlex
import subprocess

import pytz
import torch
import torchvision
import numpy as np
import yaml

import torchfcn
import matplotlib.pyplot as plt

cuda = False

root = '/media/hungnd/Data/Research/datasets'
kwargs = {'num_workers': 4, 'pin_memory': True} if cuda else {}

train_loader = torch.utils.data.DataLoader(
    torchfcn.datasets.SBDClassSeg(root, split='train', transform=True),
    batch_size=2, shuffle=True, **kwargs)

val_loader = torch.utils.data.DataLoader(
    torchfcn.datasets.VOC2011ClassSeg(
        root, split='seg11valid', transform=True),
    batch_size=2, shuffle=False, **kwargs)

for idx, (data, target) in enumerate(train_loader):
    print(idx)
    print(data.size())

    img = torchvision.utils.make_grid(data).numpy()
    img = np.transpose(img, (1, 2, 0))
    img = img[:, :, ::-1]
    plt.imshow(img)
    plt.show()

    target = target.numpy()
    target = target.squeeze()
    print(target[100:150, 100:150])
    plt.imshow(target)
    plt.show()
