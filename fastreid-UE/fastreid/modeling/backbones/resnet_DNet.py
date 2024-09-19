""" --------------------------------------------------------------------------- BA_UE
This entire file is new but copy-pasted as far as possible from resnet.py

Adapted by Andreas Gebhardt in 2024.
"""
import math

from torch import nn

from .build import BACKBONE_REGISTRY
from fastreid.layers import (
    IBN,
    SELayer,
    get_norm as get_norm_orig,
)

# we overwrite get_norm to set the momentum as it is set in DNet and call the original function
# this saves the manual editing of all the get_norm calls
def get_norm(norm, out_channels, **kwargs):
    if norm == "BN":
        return get_norm_orig(norm, out_channels, momentum=0.003, **kwargs)
    else:
        return get_norm_orig(norm, out_channels, **kwargs)


class Bottleneck_DNet(nn.Module): 
    expansion = 4

    # planes = channels
    # depth_in, depth_bottleneck, depth == inplanes, planes, outplanes
    def __init__(self, inplanes, planes, outplanes, bn_norm, with_ibn=False, with_se=False,
                 stride=1, downsample=None, reduction=16):
        super(Bottleneck_DNet, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        if with_ibn:
            self.bn1 = IBN(planes, bn_norm)
        else:
            self.bn1 = get_norm(bn_norm, planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = get_norm(bn_norm, planes)
        self.conv3 = nn.Conv2d(planes, outplanes, kernel_size=1, bias=False)
        self.bn3 = get_norm(bn_norm, outplanes)
        self.relu = nn.ReLU(inplace=True)
        if with_se:
            self.se = SELayer(outplanes, reduction)
        else:
            self.se = nn.Identity()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet_DNet(nn.Module):
    def __init__(self, bn_norm, with_ibn, with_se):
        self.inplanes = 64 # number of channels after the PreBlocks
        block = Bottleneck_DNet

        super().__init__()

        """ --------------------------------------------------------------------------- BA_UE
        PreBlocks
        """
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = get_norm(bn_norm, self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)


        # inplanes are outplanes of previous layer
        self.layer1 = self._make_layer(block, [(256, 64, 1)] * 2 + [(256, 64, 2)], 
                                        self.inplanes, bn_norm, with_ibn, with_se)

        self.layer2 = self._make_layer(block, [(512, 128, 1)] * 3 + [(512, 128, 2)], 
                                        256, bn_norm, with_ibn, with_se)

        self.layer3 = self._make_layer(block, [(1024, 256, 1)] * 5 + [(1024, 256, 2)], 
                                        512, bn_norm, with_ibn, with_se)

        self.layer4 = self._make_layer(block, [(2048, 512, 1)] * 3, 
                                        1024, bn_norm, with_se=with_se)

        # <-----------------------------------------------------------------------------------------------------------------------------------------
        #                                      
        self.random_init()


    # dds: list of tuples (depth, depth_bottleneck, stride)
    def _make_layer(self, block, dds_list, inplanes, bn_norm="BN", with_ibn=False, with_se=False):

        layers = []

        for index, dds in enumerate(dds_list):
            
            # naming as in DNet
            depth, depth_bottleneck, stride = dds

            # naming as in Fastreid
            planes = depth_bottleneck
            outplanes = depth
            
            # for the first element in each dds_list we want to add the downsample block in the skip connection
            # because that is the only instance where the channel number changes (depth != depth_in)
            # We can ignore the subsample as in this instance, stride is always 1
            downsample = None
            if index == 0:
                downsample = nn.Sequential(
                    nn.Conv2d(inplanes, outplanes,
                            kernel_size=1, stride=stride, bias=False),
                    get_norm(bn_norm, outplanes),
                )
            elif stride != 1:
                downsample = nn.MaxPool2d(kernel_size=1, stride=stride)
            
            layers.append(block(inplanes, planes, outplanes, bn_norm, with_ibn, with_se, stride, downsample))

            # since channel number between blocks is the same within one layer, 
            # we overwrite this for the rest of the layer
            inplanes = outplanes

        return nn.Sequential(*layers)

    def forward(self, x):
        # PreBlocks
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

    def random_init(self): # same as in TF: truncated variance_scaling_initializer with default parameters
        for m in self.modules():
            # print("ResNet: Initializing", m)
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels # using fan_in to conform to TF
                nn.init.trunc_normal_(m.weight, 0, math.sqrt(2 * 1.3 / n)) # added factor of 1.3 present in TF, also truncated (standard values of 2 stddev fit)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)



@BACKBONE_REGISTRY.register()
def build_resnet_DNet_backbone(cfg):
    """
    Create a ResNet instance from config that is equivalent to the one used for DNet.
    Returns:
        ResNet: a :class:`ResNet` instance.
    """

    # fmt: off
    pretrain      = cfg.MODEL.BACKBONE.PRETRAIN
    pretrain_path = cfg.MODEL.BACKBONE.PRETRAIN_PATH
    bn_norm       = cfg.MODEL.BACKBONE.NORM
    with_ibn      = cfg.MODEL.BACKBONE.WITH_IBN
    with_se       = cfg.MODEL.BACKBONE.WITH_SE
    with_nl       = cfg.MODEL.BACKBONE.WITH_NL
    # fmt: on

    model = ResNet_DNet(bn_norm, with_ibn, with_se)

    return model