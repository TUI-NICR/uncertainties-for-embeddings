import math

from torch import nn

from fastreid.config import configurable
from fastreid.layers import *
from fastreid.layers.UAL import Bayes_Dropout_Conv2d, Bayes_Gaussion_Conv2d, BayesDropoutConv2_5d, BayesGaussianConv2_5d

from ..heads.build import get_var_head_feat_dim
from .build import REID_NECKS_REGISTRY, get_neck_feat_dim

"""
@author:    Andreas Gebhardt
@contact:   AGebhardt1999@gmail.com
"""


BAYESIAN_CONV_MAP = {
    "Dropout": Bayes_Dropout_Conv2d,
    "Gaussian": Bayes_Gaussion_Conv2d,
    "Dropout2.5D": BayesDropoutConv2_5d,
    "Gaussian2.5D": BayesGaussianConv2_5d
}


@REID_NECKS_REGISTRY.register()
class UAL_neck(nn.Module):
    """
    The bayesian module from UAL.
    """

    @configurable
    def __init__(
        self, 
        *, 
        in_dim,
        mid_dims,
        out_dim,
        norm_type,
        p,
        layer_name,
        enable_3d,
        parallel_BN_full_3d 
    ) -> None:
        """
        NOTE: this interface is experimental.
        """
        super().__init__()

        # code copied from UAL and then adapted

        #BN_MOMENTUM = 0.1
        dim_list = [in_dim] + mid_dims + [out_dim]
        self.layer = nn.Sequential()
        self.bayes_count = len(dim_list) - 1
        self.bayes_index = []
        self.p = p
        # self.p = 0.95
        for i in range(self.bayes_count):
            in_dim, out_dim = dim_list[i], dim_list[i + 1]
            self.bayes_index.append(i * 3)

            self.layer.add_module(
                'Bayes_{}'.format(i),
                BAYESIAN_CONV_MAP[layer_name](
                    in_channels=in_dim, 
                    out_channels=out_dim, 
                    kernel_size=1, 
                    stride=1, 
                    padding=0, 
                    prob=self.p)
                )
            
            #self.layer.add_module('BN_{}'.format(i), nn.BatchNorm2d(out_dim, momentum=BN_MOMENTUM))
            self.layer.add_module('BN_{}'.format(i), get_norm(norm_type, out_dim, enable_3d=enable_3d, parallel_BN_full_3d=parallel_BN_full_3d))#, bias_freeze=True)) # try with this, if result suffers, change back. Momentum is the same.
            if i < self.bayes_count - 1:
                self.layer.add_module('ReLU_{}'.format(i), nn.ReLU(inplace=True))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        pass

    @classmethod
    def from_config(cls, cfg):
        in_dim = get_neck_feat_dim(cfg)
        mid_dims = cfg.MODEL.NECK.MID_DIMS
        out_dim = get_var_head_feat_dim(cfg)
        norm_type     = cfg.MODEL.HEADS.NORM
        p = cfg.MODEL.NECK.P
        layer_name = cfg.MODEL.NECK.BAYESIAN_LAYER
        enable_3d   = cfg.MODEL.HEADS.PARALLEL_3D_CONV
        parallel_BN_full_3d = cfg.MODEL.HEADS.PARALLEL_BN_FULL_3D

        return {
            'in_dim': in_dim,
            'mid_dims': mid_dims,
            'out_dim': out_dim,
            'norm_type': norm_type,
            'p': p,
            'layer_name': layer_name,
            'enable_3d': enable_3d,
            'parallel_BN_full_3d': parallel_BN_full_3d
        }

    def forward(self, features, targets=None):
        out = self.layer(features)
        return out