import torch.nn as nn
import torch.functional as F

mbox_cfg = {
    '300': [4, 6, 6, 6, 4, 4]
}

"""
multibox: generate location prediction and confidence scores
"""
def multibox(vgg, extra_layers, cfg, num_classes):
    loc_layers = []
    conf_layers = []
    vgg_source = [24, -2]

    """
    use Con4_3 and Conv7 (correspond to layer index 24 and -2) to predict
    """
    for k, v in enumerate(vgg_source):
        loc_layers.append(
            nn.Conv2d(vgg[v].out_channels, cfg[k] * 4, kernel_size=3, padding=1))
        conf_layers.append(nn.Conv2d(vgg_source[v].out_channels, cfg[k] * num_classes, kernel_size=3,
                                     padding=1))

    """
    use the second layer as feature map in every two layers
    """
    for k, v in enumerate(extra_layers[1::2], 2):
        loc_layers.append(
            nn.Conv2d(v.out_channels, cfg[k] * 4, kernel_size=3, padding=1))
        conf_layers.append(nn.Conv2d(v.out_channels, cfg[k] * num_classes, kernel_size=3,
                                     padding=1))
    return vgg, extra_layers, loc_layers, conf_layers # vgg, extra, head