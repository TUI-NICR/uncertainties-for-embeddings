from fastreid.layers import any_softmax
import torch
import torch.nn.functional as F
from torch import nn

from fastreid.config import configurable
from fastreid.layers import *
from fastreid.layers import pooling

from ..heads.build import get_mean_head_feat_dim
from .build import REID_CROWNS_REGISTRY

"""
@author:    Andreas Gebhardt
@contact:   AGebhardt1999@gmail.com
"""


@REID_CROWNS_REGISTRY.register()
class UAL_crown(nn.Module):
    """
    The crown from UAL.
    """

    @configurable
    def __init__(
        self, 
        *, 
        in_dim,
        out_dim,
        norm_type,
        enable_3d,
        parallel_BN_full_3d
    ) -> None:
        """
        NOTE: this interface is experimental.
        """
        super().__init__()

        self.enable_3d = enable_3d

        if enable_3d:
            Conv = nn.Conv3d # the 1x1 convolution will be applied to all the depth-wise slices of the volume. The filter will extend along the channel dimension same as with 2d case
            Pool = pooling.GlobalAvgPool2_5d
        else:
            Conv = nn.Conv2d
            Pool = pooling.GlobalAvgPool

        self.pool_layer = Pool()

        self.bottleneck = get_norm(norm_type, in_dim, enable_3d=enable_3d, parallel_BN_full_3d=parallel_BN_full_3d, bias_freeze=True)

        self.logit_layer = Conv(in_dim, out_dim, 1, bias=False) # FC layer

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.logit_layer.weight, std=0.01) # this is the same as in the standard embedding_head. Bias is initialized to 0 by default.

    @classmethod
    def from_config(cls, cfg):

        in_dim = get_mean_head_feat_dim(cfg)
        out_dim = cfg.MODEL.HEADS.NUM_CLASSES
        norm_type = cfg.MODEL.HEADS.NORM
        enable_3d   = cfg.MODEL.HEADS.PARALLEL_3D_CONV
        parallel_BN_full_3d = cfg.MODEL.HEADS.PARALLEL_BN_FULL_3D

        return {
            'in_dim': in_dim,
            'out_dim': out_dim,
            'norm_type': norm_type,
            'enable_3d': enable_3d,
            'parallel_BN_full_3d': parallel_BN_full_3d
        }

    def forward(self, features, uncertainty):

        # unpack inputs
        # UAL crown does not require vector form
        mu = features["mean"]
        var = uncertainty["variance"] 

        # (B,C,D,H,W)

        if (len(mu.shape) == 2 and not self.enable_3d) or (len(mu.shape) == 3 and self.enable_3d):
            mu = mu.unsqueeze(-1).unsqueeze(-1)

        if (len(mu.shape) == 2 and not self.enable_3d) or (len(mu.shape) == 3 and self.enable_3d):
            var = var.unsqueeze(-1).unsqueeze(-1)

        mu_scaled = mu / var.clamp(1e-10)

        mu_scaled_vector = self.pool_layer(mu_scaled)

        mu_scaled_vector_BN = self.bottleneck(mu_scaled_vector)
        #mu_scaled_vector_BN = mu_scaled_vector_BN[..., 0, 0] # removing height and width dimensions

        logits = self.logit_layer(mu_scaled_vector_BN) # UAL collapses before logit calculation but this is equivalent. Since it is just a matrix multiplication (which is distributive), this does not matter however. Collapsing before or after is equivalent.

        ret = {
            'logits': logits[..., 0, 0],
            'mean_vector': mu_scaled_vector[..., 0, 0],
            'mean_vector_raw': self.pool_layer(mu)[..., 0, 0],
            'variance_vector': uncertainty["variance_vector"] # already squeezed in var_head
        }

        # because the variance of means in the 3D case is calculated using the unscaled mu (which is returned as mean vector) in UAL 
        # we must do it here and can't let the meta_arch handle it (mean is calculated using the scaled version, var using the unscaled version)
        if self.enable_3d:
            # variance is only calculated after all samples have been aggregated in meta arch
            todo_variance_of_means = self.pool_layer(mu)[..., 0, 0]
            
            ret["todo_variance_of_mean_vector"] = todo_variance_of_means             # supposedly model uncertainty
        
        if not self.training:
            ret["mean_vector"] = mu_scaled_vector_BN[..., 0, 0] 
    
        return ret