# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

"""
Modified by Andreas Gebhardt in 2024.
"""

from .build import META_ARCH_REGISTRY, build_model


# import all the meta_arch, so they will be registered
from .baseline import Baseline
from .baseline_DNet import Baseline_DNet
from .mgn import MGN
from .moco import MoCo
from .distiller import Distiller
from .uncertainty import Uncertainty
from .uncertainty_UAL import Uncertainty_UAL
from .original_UAL_meta_arch import BaselineUAL