# encoding: utf-8
"""
@author:    Andreas Gebhardt
@contact:   AGebhardt1999@gmail.com
"""

""" --------------------------------------------------------------------------- BA_UE
this file is an adapted copy of embedding_head.py that adds the uncertainty estimation and extra loss
"""


import math
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions.multivariate_normal import MultivariateNormal

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
class EmbeddingHead_DNet(nn.Module):
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
            pool_type,
            scale,
            margin,
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
        self.sig_layer = nn.Conv2d(feat_dim, feat_dim, (8, 4), bias=True) # (8,4) should be the shape of the input tensor   |   maybe swap this for a proper linear layer in the poper model

        self.dropout_layer = nn.Dropout() # standard p=0.5
        self.logit_layer = nn.Conv2d(feat_dim, num_classes, 1, bias=True)

        self.softmax_layer = nn.Softmax(dim=1)

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
        embedding_dim = cfg.MODEL.HEADS.EMBEDDING_DIM # standard: 0
        num_classes   = cfg.MODEL.HEADS.NUM_CLASSES
        pool_type     = cfg.MODEL.HEADS.POOL_LAYER
        scale         = cfg.MODEL.HEADS.SCALE # default: 1
        margin        = cfg.MODEL.HEADS.MARGIN # default: 0
        norm_type     = cfg.MODEL.HEADS.NORM
        # fmt: on
        return {
            'feat_dim': feat_dim,
            'embedding_dim': embedding_dim,
            'num_classes': num_classes,
            'pool_type': pool_type,
            'scale': scale,
            'margin': margin,
            'norm_type': norm_type
        }

    # targets is unused because it is also unused in normal embedding head as long as Linear is used as cls_layer
    # also: cls_layer is identity as long as margin is set to 1 (default) and cls_type is 'Linear'
    def forward(self, features, targets=None): # for ResNet_DNet, features are something like (2048, 4, 8)
        """
        See :class:`ReIDHeads.forward`.
        """

        mu_raw = self.pool_layer(features) # mean
        mu = self.dropout_layer(mu_raw)

        sig = self.sig_layer(features) # NOT variance, it is the "scale" (stddev) (actually, covariance = diag_embed(square(softplus(sig))))
        sig += 1e-10

        # TF's tf.contrib.distributions.MultivariateNormalDiagWithSoftplusScale applies softplus
        # to the scale vector, which is the diagonal of the covariance matrix
        #dist = MultivariateNormal(mu.squeeze(), torch.diag_embed(F.softplus(sig.squeeze())), validate_args=False) 
        #sample = dist.rsample() # draw one sample using reparametrization trick
        #sample = sample.unsqueeze(-1).unsqueeze(-1) # get it to the same shape as mu

        # manual rsample
        eps = torch.empty(mu.shape, dtype=mu.dtype, device=mu.device).normal_()
        sample = mu + eps * F.softplus(sig)

        logits = self.logit_layer(mu).squeeze()
        sample_logits = self.logit_layer(sample).squeeze()
        
        return {
            "cls_outputs": logits,
            "sampled_cls_outputs": sample_logits,
            "pred_class_logits": self.softmax_layer(logits + 0.1 * sample_logits), # DNet also stores this sum under "predictions" but never uses it afaict. For eval it uses PreLogits with direct comparison
            "features": mu.squeeze(), # PreLogits_mean
            "features_sig": sig.squeeze() # PreLogits_sig
        }
        
