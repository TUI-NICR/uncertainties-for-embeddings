from torch import nn

from fastreid.config import configurable
from fastreid.layers import pooling

from .build import REID_HEADS_REGISTRY

"""
@author:    Andreas Gebhardt
@contact:   AGebhardt1999@gmail.com
"""


@REID_HEADS_REGISTRY.register()
class DNet_mean_head(nn.Module):
    """
    The mean head from DistributionNet.
    """

    @configurable
    def __init__(
        self, 
        *, 
        pool_type,
        dropout_p
    ) -> None:
        """
        NOTE: this interface is experimental.
        """
        super().__init__()

        self.pool_layer = getattr(pooling, pool_type)() # default: GlobalAvgPool
        self.dropout_layer = nn.Dropout(dropout_p) # default: 0.5
        
        self.reset_parameters()

    def reset_parameters(self) -> None:
        pass

    @classmethod
    def from_config(cls, cfg):

        pool_type = cfg.MODEL.MEAN_HEAD.POOL_TYPE
        dropout_p = cfg.MODEL.MEAN_HEAD.DROPOUT_P

        return {
            'pool_type': pool_type,
            'dropout_p': dropout_p
        }

    def forward(self, features):

        mu_raw = self.pool_layer(features)
        mu = self.dropout_layer(mu_raw).squeeze() # dropout is disabled by eval-mode, no no need to pass along mu_raw

        return {
            'mean': mu,
            'mean_vector': mu 
        }