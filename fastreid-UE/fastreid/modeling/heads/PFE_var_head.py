from ..necks.build import get_neck_feat_dim, get_neck_feat_hw
import torch
from torch import nn

from fastreid.config import configurable
from fastreid.layers import *

from .build import REID_HEADS_REGISTRY, get_var_head_feat_dim

"""
@author:    Andreas Gebhardt
@contact:   AGebhardt1999@gmail.com
"""


@REID_HEADS_REGISTRY.register()
class PFE_var_head(nn.Module):
    """
    
    """

    @configurable
    def __init__(
        self, 
        *, 
        in_dim,
        mid_dims,
        out_dim,
        in_hw,
        pooling_size,
        norm_type    
    ) -> None:
        """
        NOTE: this interface is experimental.
        """
        super().__init__()
        
        if len(mid_dims) == 0:
            mid_dim = in_dim
        else:
            mid_dim = mid_dims[0]

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

        self.fc1 = nn.Conv2d(in_dim, mid_dim, conv_in_hw)
        self.bn1 = get_norm(norm_type, mid_dim)#, bias_freeze=True)

        self.fc2 = nn.Conv2d(mid_dim, out_dim, 1)
        self.bn2 = get_norm(norm_type, out_dim)#, bias_freeze=True)

        self.relu = nn.ReLU(inplace=True)

        self.gamma = nn.Parameter(torch.Tensor(1)).to(self.fc1.weight.device)
        self.beta = nn.Parameter(torch.Tensor(1)).to(self.fc1.weight.device)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.constant_(self.gamma, 1e-4)
        nn.init.constant_(self.beta, -7.0)

        """nn.init.constant_(self.gamma, 1e-1)
        nn.init.constant_(self.beta, 0.0)"""

    @classmethod
    def from_config(cls, cfg):

        in_dim = get_neck_feat_dim(cfg)
        mid_dims = cfg.MODEL.VAR_HEAD.MID_DIMS
        out_dim = get_var_head_feat_dim(cfg)
        norm_type     = cfg.MODEL.HEADS.NORM
        in_hw = get_neck_feat_hw(cfg)
        pooling_size = cfg.MODEL.VAR_HEAD.POOLING_SIZE

        return {
            'in_dim': in_dim,
            'mid_dims': mid_dims,
            'out_dim': out_dim,
            'in_hw': in_hw,
            'pooling_size': pooling_size,
            'norm_type': norm_type
        }

    def forward(self, features, targets=None):

        pool_features = self.pool(features) # avoid OOM error and massive parameter number
        
        sig = self.fc1(pool_features)
        sig = self.bn1(sig)
        sig = self.relu(sig)

        sig = self.fc2(sig)
        sig = self.bn2(sig)

        sig = self.gamma * sig + self.beta
        variance_vector = torch.exp(sig).squeeze() + 1e-6

        return {
            "variance": variance_vector,
            "variance_vector": variance_vector
        }