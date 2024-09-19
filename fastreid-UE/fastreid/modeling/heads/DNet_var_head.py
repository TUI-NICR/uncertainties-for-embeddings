import torch
from torch import nn
import torch.nn.functional as F

from fastreid.config import configurable
from fastreid.layers.weight_init import weights_init_kaiming

from .build import REID_HEADS_REGISTRY, get_var_head_feat_dim
from ..necks.build import get_neck_feat_dim, get_neck_feat_hw

"""
@author:    Andreas Gebhardt
@contact:   AGebhardt1999@gmail.com
"""


@REID_HEADS_REGISTRY.register()
class DNet_var_head(nn.Module):
    """
    The var head from DistributionNet
    """

    @configurable
    def __init__(
        self, 
        *, 
        in_dim,
        in_hw,
        out_dim,
        pooling_size
    ) -> None:
        """
        NOTE: this interface is experimental.
        """
        super().__init__()

        # handle multiple kernel types
        if type(pooling_size) == int:
            if pooling_size == -1: # -1 = GAP
                pooling_size = in_hw
            else:
                pooling_size = (pooling_size, pooling_size)
        elif type(pooling_size) == tuple or type(pooling_size) == list:
            if len(pooling_size) != 2:
                raise TypeError(f"MODEL.VAR_HEAD.POOLING_SIZE must be int, tuple (h, w), or list [h, w] but is {pooling_size}.")
        else:
            raise TypeError(f"MODEL.VAR_HEAD.POOLING_SIZE must be int, tuple (h, w), or list [h, w] but is {pooling_size}.")

        self.pool = nn.AvgPool2d(pooling_size) 

        conv_in_hw = (in_hw[0] // pooling_size[0], in_hw[1] // pooling_size[1])
        self.conv = nn.Conv2d(in_dim, out_dim, conv_in_hw)
        
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.conv.apply(weights_init_kaiming) # this is at least similar to DNet, except using the fastreid version of the initialization

    @classmethod
    def from_config(cls, cfg):

        in_dim = get_neck_feat_dim(cfg)
        in_hw = get_neck_feat_hw(cfg)
        out_dim = get_var_head_feat_dim(cfg)

        return {
            'in_dim': in_dim,
            'in_hw': in_hw,
            'out_dim': out_dim,
            'pooling_size': cfg.MODEL.VAR_HEAD.POOLING_SIZE
        }

    def forward(self, features, targets=None):

        pool_features = self.pool(features) # to have the actual input size be the same as DNet and also avoid OOM error

        scale = self.conv(pool_features).squeeze() # in DNet this is the *scale*, NOT covariance: covariance = diag_embed(square(softplus(sig)))
        scale += 1e-10
        scale = F.softplus(scale)

        variance = torch.square(scale)

        return {
            'variance': variance,
            'variance_vector': variance
        }