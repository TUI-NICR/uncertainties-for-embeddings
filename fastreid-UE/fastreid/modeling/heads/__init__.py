# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

"""
Modified by Andreas Gebhardt in 2024.
"""

from .build import REID_HEADS_REGISTRY, build_heads, build_mean_head, build_var_head

# import all the meta_arch, so they will be registered
from .embedding_head import EmbeddingHead
from .embedding_head_DNet import EmbeddingHead_DNet
from .embedding_head_DNet_pretrain import EmbeddingHead_DNet_pretrain
from .clas_head import ClasHead

# mod_arch components
from .DNet_mean_head import DNet_mean_head
from .DNet_var_head import DNet_var_head
from .UAL_mean_head import UAL_mean_head
from .UAL_var_head import UAL_var_head
from .PFE_var_head import PFE_var_head
from .baseline_mean_head import Baseline_mean_head
from .original_UAL_head import BayesHead