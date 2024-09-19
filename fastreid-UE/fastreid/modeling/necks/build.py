# encoding: utf-8

"""
@author:    Andreas Gebhardt
@contact:   AGebhardt1999@gmail.com
"""

from ...utils.registry import Registry
import torch

REID_NECKS_REGISTRY = Registry("NECKS")
REID_NECKS_REGISTRY.__doc__ = """
Registry for reid necks in an uncertainty model.
"""

def build_neck(cfg):
    if cfg.MODEL.NECK.NAME == "Identity": # default
        return torch.nn.Identity()
    else:
        return REID_NECKS_REGISTRY.get(cfg.MODEL.NECK.NAME)(cfg)
    
def get_neck_feat_dim(cfg):
    
    if cfg.MODEL.NECK.FEAT_DIM != None:
        return cfg.MODEL.NECK.FEAT_DIM
    
    else:
        return cfg.MODEL.BACKBONE.FEAT_DIM
    

def get_neck_feat_hw(cfg):
    
    if cfg.MODEL.NECK.FEAT_HW != None:
        return cfg.MODEL.NECK.FEAT_HW
    
    else:
        return cfg.MODEL.BACKBONE.FEAT_HW
    