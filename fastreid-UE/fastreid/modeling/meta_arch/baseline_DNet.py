# encoding: utf-8
"""
@author:    Andreas Gebhardt
@contact:   AGebhardt1999@gmail.com
"""

""" --------------------------------------------------------------------------- BA_UE
copy of baseline to include DNet-specific stuff
"""

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

import logging
logger = logging.getLogger(__name__)

from fastreid.utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message

from fastreid.config import configurable
from fastreid.modeling.backbones import build_backbone
from fastreid.modeling.heads import build_heads
from fastreid.modeling.losses import *
from .build import META_ARCH_REGISTRY

from fastreid.utils.BA_UE_utils import getattr_rec, get_state_dict_from_TF_checkpoint


ENTROPY_CONSTANT = 2048*(np.log(2*np.pi) + 1)/2 # no need to calculate this in every iteration
ENTROPY_THRESHOLD = np.log(5) + (1+ np.log(2*np.pi))/2

@META_ARCH_REGISTRY.register()
class Baseline_DNet(nn.Module):
    """
    based on Baseline architecture, Any models that contains the following two components:
    1. Per-image feature extraction (aka backbone)
    2. Per-image feature aggregation and loss computation
    """

    @configurable
    def __init__(
            self,
            *,
            backbone,
            heads,
            pixel_mean,
            pixel_std,
            loss_kwargs=None,
            pretrain_path,
            modules_to_freeze
    ):
        """
        NOTE: this interface is experimental.

        Args:
            backbone:
            heads:
            pixel_mean:
            pixel_std:
        """
        super().__init__()
        # backbone
        self.backbone = backbone

        # head
        self.heads = heads

        self.loss_kwargs = loss_kwargs

        self.register_buffer('pixel_mean', torch.Tensor(pixel_mean).view(1, -1, 1, 1), False)
        self.register_buffer('pixel_std', torch.Tensor(pixel_std).view(1, -1, 1, 1), False)

        incompatible = None
        # load TF pretrain
        if pretrain_path != '' and pretrain_path[0] == "§":
            pretrain_path = pretrain_path[1:]

            logger.info("Reading TensorFlow Checkpoint...")
            state_dict = get_state_dict_from_TF_checkpoint(pretrain_path, logger)

            logger.info("Loading State Dict...")
            incompatible = self.load_state_dict(state_dict, strict=False)

            if incompatible.missing_keys:
                logger.info(
                    get_missing_parameters_message(incompatible.missing_keys)
                )
            if incompatible.unexpected_keys:
                logger.info(
                    get_unexpected_parameters_message(incompatible.unexpected_keys)
                )

        self.sampled_CE_loss = True # is set to False when the model realizes that there is no sampling going on. controls processing of sampled FV

        """ --------------------------------------------------------------------------- BA_UE
        freeze weights of modules specified in config FREEZE_WEIGTHS
        """
        for module_name in modules_to_freeze:
            module = getattr_rec(self, module_name)

            # set module and all of its submodules to eval mode (-> BatchNorm, Dropout, ...)
            module.eval()
            module.train = lambda self, mode=True: self # permanently set to eval mode

            # it is necessary to set this manually since eval() does not affect requires_grad
            for param in module.parameters():
                param.requires_grad_(False)
            

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        heads = build_heads(cfg)
        pretrain_path = cfg.MODEL.BACKBONE.PRETRAIN_PATH
        modules_to_freeze = cfg.MODEL.FREEZE_WEIGHTS

        return {
            'backbone': backbone,
            'heads': heads,
            'pixel_mean': cfg.MODEL.PIXEL_MEAN,
            'pixel_std': cfg.MODEL.PIXEL_STD,
            'loss_kwargs':
                {
                    # loss name
                    'loss_names': cfg.MODEL.LOSSES.NAME,

                    # loss hyperparameters
                    'ce': {
                        'eps': cfg.MODEL.LOSSES.CE.EPSILON,
                        'alpha': cfg.MODEL.LOSSES.CE.ALPHA,
                        'scale': cfg.MODEL.LOSSES.CE.SCALE
                    },
                    'tri': {
                        'margin': cfg.MODEL.LOSSES.TRI.MARGIN,
                        'norm_feat': cfg.MODEL.LOSSES.TRI.NORM_FEAT,
                        'hard_mining': cfg.MODEL.LOSSES.TRI.HARD_MINING,
                        'scale': cfg.MODEL.LOSSES.TRI.SCALE
                    },
                    'circle': {
                        'margin': cfg.MODEL.LOSSES.CIRCLE.MARGIN,
                        'gamma': cfg.MODEL.LOSSES.CIRCLE.GAMMA,
                        'scale': cfg.MODEL.LOSSES.CIRCLE.SCALE
                    },
                    'cosface': {
                        'margin': cfg.MODEL.LOSSES.COSFACE.MARGIN,
                        'gamma': cfg.MODEL.LOSSES.COSFACE.GAMMA,
                        'scale': cfg.MODEL.LOSSES.COSFACE.SCALE
                    }
                },
            'pretrain_path': pretrain_path,
            'modules_to_freeze': modules_to_freeze
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images)

        if self.training:
            assert "targets" in batched_inputs, "Person ID annotation are missing in training!"
            targets = batched_inputs["targets"]

            # PreciseBN flag, When do preciseBN on different dataset, the number of classes in new dataset
            # may be larger than that in the original dataset, so the circle/arcface will
            # throw an error. We just set all the targets to 0 to avoid this problem.
            if targets.sum() < 0: targets.zero_()

            outputs = self.heads(features, targets)
            losses = self.losses(outputs, targets)
            return losses
        else:
            out = self.heads(features)
            outputs = {
                "mean": out["features"],
                "variance": out["features_sig"]
            }
            return outputs

    def preprocess_image(self, batched_inputs):
        """
        Normalize and batch the input images.
        """
        if isinstance(batched_inputs, dict):
            images = batched_inputs['images']
        elif isinstance(batched_inputs, torch.Tensor):
            images = batched_inputs
        else:
            raise TypeError("batched_inputs must be dict or torch.Tensor, but get {}".format(type(batched_inputs)))

        images.sub_(self.pixel_mean).div_(self.pixel_std)
        return images

    def losses(self, outputs, gt_labels):
        """
        Compute loss from modeling's outputs, the loss function input arguments
        must be the same as the outputs of the model forwarding.
        """
        # model predictions
        # fmt: off
        pred_class_logits = outputs['pred_class_logits'].detach()
        cls_outputs       = outputs['cls_outputs']
        if self.sampled_CE_loss:
            try:
                sampled_cls_outputs       = outputs['sampled_cls_outputs']
                sigma = outputs['features_sig']
            except KeyError:
                self.sampled_CE_loss = False
        pred_features     = outputs['features']
        # fmt: on

        # Log prediction accuracy
        log_accuracy(pred_class_logits, gt_labels)

        loss_dict = {}
        loss_names = self.loss_kwargs['loss_names']

        if 'CrossEntropyLoss' in loss_names:
            ce_kwargs = self.loss_kwargs.get('ce')
            loss_dict['loss_cls'] = cross_entropy_loss(
                cls_outputs,
                gt_labels,
                ce_kwargs.get('eps'),
                ce_kwargs.get('alpha')
            ) * ce_kwargs.get('scale')

        if 'TripletLoss' in loss_names:
            tri_kwargs = self.loss_kwargs.get('tri')
            loss_dict['loss_triplet'] = triplet_loss(
                pred_features,
                gt_labels,
                tri_kwargs.get('margin'),
                tri_kwargs.get('norm_feat'),
                tri_kwargs.get('hard_mining')
            ) * tri_kwargs.get('scale')

        if 'CircleLoss' in loss_names:
            circle_kwargs = self.loss_kwargs.get('circle')
            loss_dict['loss_circle'] = pairwise_circleloss(
                pred_features,
                gt_labels,
                circle_kwargs.get('margin'),
                circle_kwargs.get('gamma')
            ) * circle_kwargs.get('scale')

        if 'Cosface' in loss_names:
            cosface_kwargs = self.loss_kwargs.get('cosface')
            loss_dict['loss_cosface'] = pairwise_cosface(
                pred_features,
                gt_labels,
                cosface_kwargs.get('margin'),
                cosface_kwargs.get('gamma'),
            ) * cosface_kwargs.get('scale')

        if self.sampled_CE_loss:
            # just hardcode the loss because this is not meant to be pretty, just to work. pretty is for the modular architecture
            loss_dict['loss_cls_dnet'] = cross_entropy_loss(
                    sampled_cls_outputs, 
                    gt_labels,
                    ce_kwargs.get('eps'),
                    ce_kwargs.get('alpha')
                ) * 0.1
            
            entropy_loss_avg = 0
            for sig in sigma: # iterate over batch
                # softplus scale is used in TF
                entropy = 0.5*torch.sum(torch.log(torch.square(F.softplus(sig)) + 1e-20)) + ENTROPY_CONSTANT  # add softplus since TF uses it and add epsilon=1e-20 to avoid infinity
                entropy_loss_avg += F.relu(ENTROPY_THRESHOLD - entropy/2048) # division by 2048 is present in TF code but not explained in paper

            entropy_loss_avg = entropy_loss_avg / sigma.shape[0] # average over batch

            loss_dict['loss_FU_dnet'] = entropy_loss_avg * 0.001 # this is probably what TF is doing in a very complicated way

        return loss_dict