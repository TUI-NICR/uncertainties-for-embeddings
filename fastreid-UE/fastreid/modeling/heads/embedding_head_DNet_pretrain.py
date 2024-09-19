# encoding: utf-8
"""
@author:    Andreas Gebhardt
@contact:   AGebhardt1999@gmail.com
"""

import math
import torch
import torch.nn.functional as F
from torch import nn

from fastreid.config import configurable
from fastreid.layers import *
from fastreid.layers import pooling, any_softmax
from fastreid.layers.weight_init import weights_init_kaiming
from .build import REID_HEADS_REGISTRY

from fastreid.layers import get_norm as get_norm_orig

# we overwrite get_norm to set the momentum as it is set in DNet and call the original function
# this saves the manual editing of all the get_norm calls
def get_norm(norm, out_channels, **kwargs):
    if norm == "BN":
        return get_norm_orig(norm, out_channels, momentum=0.003, **kwargs)
    else:
        return get_norm_orig(norm, out_channels, **kwargs)

@REID_HEADS_REGISTRY.register()
class EmbeddingHead_DNet_pretrain(nn.Module):
    """
    EmbeddingHead perform all feature aggregation in an embedding task, such as reid, image retrieval
    and face recognition

    It typically contains logic to

    1. feature aggregation via global average pooling and generalized mean pooling
    2. (optional) batchnorm, dimension reduction and etc.
    2. (in training only) margin-based softmax logits computation
    """

    @configurable
    def __init__(
            self,
            *,
            feat_dim,
            embedding_dim,
            num_classes,
            neck_feat,
            pool_type,
            cls_type,
            scale,
            margin,
            with_bnneck,
            norm_type
    ):
        """
        NOTE: this interface is experimental.

        Args:
            feat_dim:
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

        self.neck_feat = neck_feat

        self.logit_layer = nn.Conv2d(feat_dim, num_classes, 1, bias=True) # though I can't find the bias in the code, it is present in the chekpoint.

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels # using fan_in to conform to TF
                nn.init.trunc_normal_(m.weight, 0, math.sqrt(2 * 1.3 / n)) # added factor of 1.3 present in TF, also truncated (standard values of 2 stddev fit)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    @classmethod
    def from_config(cls, cfg):
        # fmt: off
        feat_dim      = cfg.MODEL.BACKBONE.FEAT_DIM
        embedding_dim = cfg.MODEL.HEADS.EMBEDDING_DIM
        num_classes   = cfg.MODEL.HEADS.NUM_CLASSES
        neck_feat     = cfg.MODEL.HEADS.NECK_FEAT
        pool_type     = cfg.MODEL.HEADS.POOL_LAYER
        cls_type      = cfg.MODEL.HEADS.CLS_LAYER
        scale         = cfg.MODEL.HEADS.SCALE
        margin        = cfg.MODEL.HEADS.MARGIN
        with_bnneck   = cfg.MODEL.HEADS.WITH_BNNECK
        norm_type     = cfg.MODEL.HEADS.NORM
        # fmt: on
        return {
            'feat_dim': feat_dim,
            'embedding_dim': embedding_dim,
            'num_classes': num_classes,
            'neck_feat': neck_feat,
            'pool_type': pool_type,
            'cls_type': cls_type,
            'scale': scale,
            'margin': margin,
            'with_bnneck': with_bnneck,
            'norm_type': norm_type
        }

    def forward(self, features, targets=None):
        """
        See :class:`ReIDHeads.forward`.
        """
        pool_feat = self.pool_layer(features)

        # Evaluation
        # fmt: off
        #if not self.training: return pool_feat.squeeze()
        # fmt: on

        logits = self.logit_layer(pool_feat)

        return {
            "cls_outputs": logits.squeeze(),
            "pred_class_logits": F.softmax(logits.squeeze(), dim=1), # careful: there should be softmax applied to this if used with Linear
            "features": pool_feat.squeeze(),
            "features_sig": torch.zeros(0) # to avoid error in baseline_DNet
        }
