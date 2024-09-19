# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

"""
Modified by Andreas Gebhardt in 2024.
"""

from ...utils.registry import Registry

REID_HEADS_REGISTRY = Registry("HEADS")
REID_HEADS_REGISTRY.__doc__ = """
Registry for reid heads in a baseline model.

ROIHeads take feature maps and region proposals, and
perform per-region computation.
The registered object will be called with `obj(cfg, input_shape)`.
The call is expected to return an :class:`ROIHeads`.
"""


def build_heads(cfg):
    """
    Build REIDHeads defined by `cfg.MODEL.REID_HEADS.NAME`.
    """
    head = cfg.MODEL.HEADS.NAME
    return REID_HEADS_REGISTRY.get(head)(cfg)

def build_mean_head(cfg):
    return REID_HEADS_REGISTRY.get(cfg.MODEL.MEAN_HEAD.NAME)(cfg)

def build_var_head(cfg):
    return REID_HEADS_REGISTRY.get(cfg.MODEL.VAR_HEAD.NAME)(cfg)

def get_mean_head_feat_dim(cfg):
    if cfg.MODEL.MEAN_HEAD.FEAT_DIM != None:
        return cfg.MODEL.MEAN_HEAD.FEAT_DIM
    
    elif cfg.MODEL.NECK.FEAT_DIM != None:
        return cfg.MODEL.NECK.FEAT_DIM
    
    else:
        return cfg.MODEL.BACKBONE.FEAT_DIM
    
def get_var_head_feat_dim(cfg):
    if cfg.MODEL.VAR_HEAD.FEAT_DIM != None:
        return cfg.MODEL.VAR_HEAD.FEAT_DIM
    
    elif cfg.MODEL.NECK.FEAT_DIM != None:
        return cfg.MODEL.NECK.FEAT_DIM
    
    else:
        return cfg.MODEL.BACKBONE.FEAT_DIM