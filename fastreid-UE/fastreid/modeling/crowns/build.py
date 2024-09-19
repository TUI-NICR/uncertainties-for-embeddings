# encoding: utf-8

from ...utils.registry import Registry
import torch

"""
@author:    Andreas Gebhardt
@contact:   AGebhardt1999@gmail.com
"""

REID_CROWNS_REGISTRY = Registry("CROWNS")
REID_CROWNS_REGISTRY.__doc__ = """
Registry for reid crowns in an uncertainty model.
"""

def build_crown(cfg):
    if cfg.MODEL.CROWN.NAME == "Identity": # default
        return torch.nn.Identity()
    else:
        return REID_CROWNS_REGISTRY.get(cfg.MODEL.CROWN.NAME)(cfg)
    

def get_crown_feat_dim(cfg):
    if cfg.MODEL.CROWN.FEAT_DIM != None:
        return cfg.MODEL.CROWN.FEAT_DIM

    elif cfg.MODEL.MEAN_HEAD.FEAT_DIM != None:
        return cfg.MODEL.MEAN_HEAD.FEAT_DIM
    
    elif cfg.MODEL.NECK.FEAT_DIM != None:
        return cfg.MODEL.NECK.FEAT_DIM
    
    else:
        return cfg.MODEL.BACKBONE.FEAT_DIM