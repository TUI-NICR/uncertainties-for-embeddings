from torch import nn

from fastreid.config import configurable
from fastreid.layers import *
from fastreid.layers import pooling
from ..necks.build import get_neck_feat_dim

from .build import REID_HEADS_REGISTRY, get_mean_head_feat_dim

"""
@author:    Andreas Gebhardt
@contact:   AGebhardt1999@gmail.com
"""


@REID_HEADS_REGISTRY.register()
class UAL_mean_head(nn.Module):
    """
    
    """

    @configurable
    def __init__(
        self, 
        *,
        in_dim,
        mid_dims,
        out_dim,
        norm_type,
        enable_3d,
        parallel_BN_full_3d
    ) -> None:
        """
        NOTE: this interface is experimental.
        """
        super().__init__()

        # code literally copied from UAL
        BN_MOMENTUM = 0.1

        if enable_3d:
            Conv = nn.Conv3d # the 1x1 convolution will be applied to all the depth-wise slices of the volume. The filter will extend along the channel dimension same as with 2d case
            Pool = pooling.GlobalAvgPool2_5d
        else:
            Conv = nn.Conv2d 
            Pool = pooling.GlobalAvgPool

        dim_list = [in_dim] + mid_dims + [out_dim]

        self.layer = nn.Sequential()

        self.bayes_count = len(dim_list) - 1 # no idea why this is called bayes_count, maybe copy/pasta error?
        for i in range(self.bayes_count):
            in_dim, out_dim = dim_list[i], dim_list[i + 1]

            self.layer.add_module('Conv_{}'.format(i),
                                  Conv(in_channels=in_dim, out_channels=out_dim, kernel_size=1,
                                            stride=1, padding=0))
            #self.layer.add_module('BN_{}'.format(i), nn.BatchNorm2d(out_dim, momentum=BN_MOMENTUM))
            self.layer.add_module('BN_{}'.format(i), get_norm(norm_type, out_dim, enable_3d=enable_3d, parallel_BN_full_3d=parallel_BN_full_3d))#, bias_freeze=True)) # try with this, if result suffers, change back. Momentum is the same.
            if i < self.bayes_count - 1:
                self.layer.add_module('ReLU_{}'.format(i), nn.ReLU(inplace=True))

        self.pool = Pool()

        self.reset_parameters()

    def reset_parameters(self) -> None:
        pass

    @classmethod
    def from_config(cls, cfg):
        in_dim      = get_neck_feat_dim(cfg)
        mid_dims    = cfg.MODEL.MEAN_HEAD.MID_DIMS
        out_dim     = get_mean_head_feat_dim(cfg)
        norm_type   = cfg.MODEL.HEADS.NORM
        enable_3d   = cfg.MODEL.HEADS.PARALLEL_3D_CONV
        parallel_BN_full_3d = cfg.MODEL.HEADS.PARALLEL_BN_FULL_3D

        return {
            'in_dim': in_dim,
            'mid_dims': mid_dims,
            'out_dim': out_dim,
            'norm_type': norm_type,
            'enable_3d': enable_3d,
            'parallel_BN_full_3d': parallel_BN_full_3d
        }

    def forward(self, features):
        mu = self.layer(features) # (B,C,D,H,W)

        mu_vector = self.pool(mu)[..., 0, 0] # (B,C,D,1,1)

        return {
            "mean": mu,
            "mean_vector": mu_vector
        }