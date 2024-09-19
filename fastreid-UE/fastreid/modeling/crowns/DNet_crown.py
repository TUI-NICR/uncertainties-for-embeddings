from fastreid.layers.pooling import GlobalAvgPool
import torch
from torch import nn
import torch.nn.functional as F

from fastreid.config import configurable
from fastreid.layers import *
from ..heads.build import get_mean_head_feat_dim

from .build import REID_CROWNS_REGISTRY

"""
@author:    Andreas Gebhardt
@contact:   AGebhardt1999@gmail.com
"""


@REID_CROWNS_REGISTRY.register()
class DNet_crown(nn.Module):
    """
    The crown from DistributionNet.
    """

    @configurable
    def __init__(
        self, 
        *, 
        in_dim,
        out_dim,
        with_bn,
        norm_type,
        num_samples,
        use_vector_features
    ) -> None:
        """
        NOTE: this interface is experimental.
        """
        super().__init__()

        if with_bn:
            self.norm = get_norm(norm_type, in_dim, bias_freeze=True)
        else:
            self.norm = nn.Identity()

        self.pooling_layer = GlobalAvgPool()

        #self.logit_layer = nn.Linear(in_dim, out_dim, bias=True)
        self.logit_layer = nn.Conv2d(in_dim, out_dim, 1, bias=True) # Linear Layer as 1x1-Conv

        self.num_samples = num_samples
        self.use_vector_features = use_vector_features

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.logit_layer.weight, std=0.01) # this is the same as in the standard embedding_head. Bias is initialized to 0 by default.

    @classmethod
    def from_config(cls, cfg):

        in_dim = get_mean_head_feat_dim(cfg)
        out_dim = cfg.MODEL.HEADS.NUM_CLASSES
        with_bn = cfg.MODEL.CROWN.WITH_BN
        norm_type = cfg.MODEL.CROWN.NORM_TYPE
        num_samples = cfg.MODEL.CROWN.NUM_SAMPLES
        use_vector_features = cfg.MODEL.CROWN.USE_VECTOR_FEATURES

        return {
            'in_dim': in_dim,
            'out_dim': out_dim,
            'with_bn': with_bn,
            'norm_type': norm_type,
            'num_samples': num_samples,
            'use_vector_features': use_vector_features
        }

    def forward(self, features, uncertainty):

        # unpack inputs
        if self.use_vector_features:
            mu = features["mean_vector"][:,:,None,None]
            var = uncertainty["variance_vector"][:,:,None,None] # shape (B, C, 1, 1)
        else:
            mu = features["mean"]
            var = uncertainty["variance"]

        logits = self.logit_layer(self.norm(self.pooling_layer(mu)))[..., 0, 0] # using GAP on 1x1 features does nothing

        sample_logits = []

        std = torch.sqrt(var)

        for _ in range(self.num_samples):
            # manually sample from the distribution using reparametrization trick
            eps = torch.empty(mu.shape, dtype=mu.dtype, device=mu.device).normal_()
            sample = mu + eps * std

            sample_logits.append(self.logit_layer(self.norm(self.pooling_layer(sample)))[..., 0, 0]) # (B, C', 1, 1) -> (B, C')
        
        return { 
            'logits': logits,
            'sample_logits': sample_logits,
            'mean_vector': features["mean_vector"],
            'variance_vector': uncertainty["variance_vector"]
        }