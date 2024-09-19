# encoding: utf-8
"""
@author:    Andreas Gebhardt
@contact:   AGebhardt1999@gmail.com
"""
from torch import nn

from fastreid.config import configurable
from fastreid.layers import pooling, get_norm
from .build import REID_HEADS_REGISTRY, get_mean_head_feat_dim


@REID_HEADS_REGISTRY.register()
class Baseline_mean_head(nn.Module):
    """
    This is just the feature vector generation part of EmbeddingHead. basically just a GAP.
    """

    @configurable
    def __init__(
            self,
            *,
            out_dim,
            norm_type,
            pool_type
    ):
        """
        NOTE: this interface is experimental.

        Args:
            out_dim:
            embedding_dim:
            num_classes:
            neck_feat:
            pool_type:
            cls_type:
            scale:
            margin:
            with_bnneck:
            norm_type:
        """
        super().__init__()

        # Pooling layer
        assert hasattr(pooling, pool_type), "Expected pool types are {}, " \
                                            "but got {}".format(pooling.__all__, pool_type)
        self.pool_layer = getattr(pooling, pool_type)()

        self.bn = get_norm(norm_type, out_dim, bias_freeze=True) 

        self.reset_parameters()

    def reset_parameters(self) -> None:
        pass

    @classmethod
    def from_config(cls, cfg):
        # fmt: off
        pool_type     = cfg.MODEL.MEAN_HEAD.POOL_TYPE
        out_dim     = get_mean_head_feat_dim(cfg)
        norm_type   = cfg.MODEL.HEADS.NORM
        # fmt: on
        return {
            'out_dim': out_dim,
            'norm_type': norm_type,
            'pool_type': pool_type
        }

    def forward(self, features, targets=None):
        """
        See :class:`ReIDHeads.forward`.
        """
        pool_feat = self.pool_layer(features)
        norm_feat = self.bn(pool_feat)[..., 0, 0]
        
        return {
            "mean": norm_feat,
            "mean_vector": norm_feat
        }
