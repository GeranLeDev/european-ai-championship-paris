"""TrustMark helpers: dilated (a trous) copy of the decoder ResNet-50, exact on the stride-32 sub-grid."""
import copy
import torch
import torch.nn as nn


def dilate_resnet(net, os=8):
    """Deep copy of a torchvision ResNet with output stride `os` (8 or 16): stride-2 convs of layer3/4 -> stride 1,
    later 3x3 convs dilated (torchvision's replace_stride_with_dilation scheme, exact a trous)."""
    n = copy.deepcopy(net)
    layers = {8: ("layer3", "layer4"), 16: ("layer4",), 32: ()}[os]
    dil = 1
    for name in ("layer3", "layer4"):
        layer = getattr(n, name)
        if name not in layers:
            continue
        prev = dil; dil *= 2
        for bi, blk in enumerate(layer):
            d = prev if bi == 0 else dil
            blk.conv2.stride = (1, 1); blk.conv2.dilation = (d, d); blk.conv2.padding = (d, d)
            if blk.downsample is not None:
                blk.downsample[0].stride = (1, 1)
    return n


def resnet_feats(n, x):
    f = n.maxpool(n.relu(n.bn1(n.conv1(x)))); out = []
    for k in ("layer1", "layer2", "layer3", "layer4"):
        f = getattr(n, k)(f); out.append(f)
    return out
