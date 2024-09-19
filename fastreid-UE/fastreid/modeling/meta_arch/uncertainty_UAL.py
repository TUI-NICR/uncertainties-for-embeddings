# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
""" --------------------------------------------------------------------------- BA_UE
This file is based on meta_arch/baseline.py but has been modified for this project.
"""


import torch
from torch import nn

from fastreid.config import configurable
from fastreid.modeling.backbones import build_backbone
from fastreid.modeling.necks import build_neck
from fastreid.modeling.heads import build_mean_head, build_var_head
from fastreid.modeling.crowns import build_crown
from fastreid.modeling.crowns.build import get_crown_feat_dim
from fastreid.modeling.losses import *
from .build import META_ARCH_REGISTRY

from fastreid.utils.BA_UE_utils import getattr_rec


@META_ARCH_REGISTRY.register()
class Uncertainty_UAL(nn.Module):
    """
    Architecture for Uncertainty Estimation. Models can contain the following components:
    - Backbone: The main feature extractor, e.g. ResNet50. This is required.
    - Neck: An intermediate module between the backbone and heads, that may be used to add
            additional processing, e.g. UAL's Bayesian Module.
    - Mean Head: The head that outputs the mean for a probabilistic embedding. This is 
            equivalent to the deterministic Re-ID head. It takes the neck output as input.
    - Var Head: The head that outputs the variance (diagonal of covariance or scale 
            depending on implementation). It takes the neck output as input.
    - Crown: A post-processing module that takes the mean and var outputs, e.g. for
            UAL's weighting of the mean using the var.
    The outputs for the loss calculation contain the outputs of mean head, var head, and crown.
    """

    @configurable 
    def __init__(
            self,
            *,
            backbone,
            neck,
            mean_head,
            var_head,
            crown,
            modules_to_freeze,
            enable_3d,
            parallel_dim_len,
            pixel_mean,
            pixel_std,
            loss_kwargs=None
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

        self.enable_3d = enable_3d
        self.parallel_dim_len = parallel_dim_len
        
        # set up components
        self.backbone = backbone
        self.neck = neck
        self.mean_head = mean_head
        self.var_head = var_head
        self.crown = crown

        
        # set up center loss
        if 'CenterLoss' in loss_kwargs['loss_names']:
            self.center_loss = CenterLoss(num_classes=loss_kwargs["center"]["num_classes"], feat_dim=loss_kwargs["center"]["feat_dim"])

        """ --------------------------------------------------------------------------- BA_UE
        freeze weights of modules specified in config FREEZE_WEIGTHS
        """
        for module_name in modules_to_freeze:
            module = getattr_rec(self, module_name)

            # set module and all of its submodules to eval mode (-> BatchNorm, Dropout, ...)
            module.eval()

            # it is necessary to set this manually since eval() does not affect requires_grad
            for param in module.parameters():
                param.requires_grad_(False)


        self.loss_kwargs = loss_kwargs

        self.register_buffer('pixel_mean', torch.Tensor(pixel_mean).view(1, -1, 1, 1), False)
        self.register_buffer('pixel_std', torch.Tensor(pixel_std).view(1, -1, 1, 1), False)

    @classmethod
    def from_config(cls, cfg):

        modules_to_freeze = cfg.MODEL.FREEZE_WEIGHTS
        enable_3d = cfg.MODEL.HEADS.PARALLEL_3D_CONV
        parallel_dim_len = cfg.MODEL.HEADS.PARALLEL_DIM_LEN

        backbone  = build_backbone(cfg)
        neck      = build_neck(cfg)
        mean_head = build_mean_head(cfg)
        var_head  = build_var_head(cfg)
        crown     = build_crown(cfg)

        return {
            'backbone': backbone,
            'neck': neck,
            'mean_head': mean_head,
            'var_head': var_head,
            'crown': crown,
            'modules_to_freeze': modules_to_freeze,
            'enable_3d': enable_3d,
            'parallel_dim_len': parallel_dim_len,
            'pixel_mean': cfg.MODEL.PIXEL_MEAN,
            'pixel_std': cfg.MODEL.PIXEL_STD,
            'loss_kwargs': # TODO: extract this unpacking to a util function
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
                        'scale': cfg.MODEL.LOSSES.TRI.SCALE,
                        'metric': cfg.MODEL.LOSSES.TRI.METRIC
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
                    },
                    'center': {
                        'scale': cfg.MODEL.LOSSES.CENTER.SCALE,
                        'num_classes': cfg.MODEL.HEADS.NUM_CLASSES,
                        'feat_dim': get_crown_feat_dim(cfg)
                    },
                    'sce': {
                        'eps': cfg.MODEL.LOSSES.SCE.EPSILON,
                        'alpha': cfg.MODEL.LOSSES.SCE.ALPHA,
                        'scale': cfg.MODEL.LOSSES.SCE.SCALE
                    },
                    'fu': {
                        'scale': cfg.MODEL.LOSSES.FU.SCALE,
                        'feat_dim': cfg.MODEL.BACKBONE.FEAT_DIM
                    },
                    'dul': {
                        'eps': cfg.MODEL.LOSSES.DUL.EPSILON,
                        'lambda': cfg.MODEL.LOSSES.DUL.LAMBDA,
                        'alpha': cfg.MODEL.LOSSES.DUL.ALPHA,
                        'scale': cfg.MODEL.LOSSES.DUL.SCALE
                    }
                }
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        
        images = self.preprocess_image(batched_inputs)
        
        deep_features = self.backbone(images)

        if self.enable_3d:
            # repeat along new dimension to compute with multiple weight samples of neck in parallel
            deep_features = deep_features.unsqueeze(2).expand(-1, -1, self.parallel_dim_len, -1, -1) 

        if self.training:
            features_neck = self.neck(deep_features)

            features = self.mean_head(features_neck) # mean vector should be located at features["mean_vector"], raw mean data is at features["mean"] (might be the same)
            uncertainty = self.var_head(features_neck) # variance vector should be located at uncertainty["variance_vector"], raw variance data is at uncertainty["variance"] (might be the same)

            # crown feats should always include logits, mean_vector, and variance_vector
            crown_feats = self.crown(features, uncertainty) 

            if self.enable_3d: # collapse previously added dimension
                
                mean_vectors     = crown_feats["mean_vector"]     # shape: (B, C, D)
                variance_vectors = crown_feats["variance_vector"] # shape: (B, C, D)
                logits_vectors   = crown_feats["logits"]          # shape: (B, C', D)

                if len(mean_vectors.shape) == 3: # don't do anything if crown handled this already
                    
                    mean_of_means = mean_vectors.mean(dim=2)
                    variance_of_means = mean_vectors.var(dim=2)
                    
                    crown_feats["mean_vector"] = mean_of_means                                              # feature vector
                    crown_feats["variance_of_mean"] = variance_of_means                                     # supposedly model uncertainty #  TODO: can we use this to scale learning rate?

                if len(variance_vectors.shape) == 3: # don't do anything if crown handled this already
                    
                    mean_of_variances = variance_vectors.mean(dim=2)
                    variance_of_variances = variance_vectors.var(dim=2) # shapes: (B, C) (batch of vectors)
                    
                    crown_feats["variance_vector"] = mean_of_variances                                      # supposedly data uncertainty
                    crown_feats["variance_of_variance"] = variance_of_variances                             # could this be (coaxed to be) distributional uncertainty? # TODO: add loss and train samples to make this happen?

                if len(logits_vectors.shape) == 3: # don't do anything if crown handled this already
                    
                    mean_of_logits = logits_vectors.mean(dim=2)
                    variance_of_logits = logits_vectors.var(dim=2) # shapes: (B, C')

                    crown_feats["logits"] = mean_of_logits
                    crown_feats["variance_of_logits"] = variance_of_logits
                
        

        
            assert "targets" in batched_inputs, "Person ID annotation are missing in training!"
            targets = batched_inputs["targets"]

            # PreciseBN flag, When do preciseBN on different dataset, the number of classes in new dataset
            # may be larger than that in the original dataset, so the circle/arcface will
            # throw an error. We just set all the targets to 0 to avoid this problem.
            if targets.sum() < 0: targets.zero_()

            # outputs = self.heads(features, targets) # why do they need targets for this?? omitting this in my own implementation for now
            losses = self.losses(crown_feats, targets)
            return losses
        else:
            
            mean_vectors = []
            variance_vectors = []

            for i in range(30):
                # don't need to do the load_bayes_model stuff since it just samples a new weigth in each forward anyway
                features_neck = self.neck(deep_features)

                features = self.mean_head(features_neck) # mean vector should be located at features["mean_vector"], raw mean data is at features["mean"] (might be the same)
                uncertainty = self.var_head(features_neck) # variance vector should be located at uncertainty["variance_vector"], raw variance data is at uncertainty["variance"] (might be the same)

                # crown feats should always include logits, mean_vector, and variance_vector
                crown_feats = self.crown(features, uncertainty) 

                mean_vectors.append(crown_feats["mean_vector"].unsqueeze(dim=0))
                variance_vectors.append(crown_feats["variance_vector"].unsqueeze(dim=0)) 

            
            outputs = {
                "mean": torch.cat(mean_vectors, dim=0).mean(dim=0),
                "variance": torch.cat(variance_vectors, dim=0).mean(dim=0),
                "variance_of_mean": torch.cat(mean_vectors, dim=0).var(dim=0)
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

        # unpacking outputs
        # we always get these
        mean_vector = outputs["mean_vector"]
        variance_vector = outputs["variance_vector"]
        logits = outputs["logits"]

        #print(f"average batch variance l2 norm: {variance_vector.norm(dim=1).mean(dim=0)}")

        # we sometimes get these
        try: # TODO: would therre be a performance increase if I made this part conditional?
            sample_logits = outputs["sample_logits"]
            sample_logits_present = True
        except KeyError:
            sample_logits_present = False


        # Log prediction accuracy
        log_accuracy(logits.detach(), gt_labels)


        loss_dict = {}
        loss_names = self.loss_kwargs['loss_names']

        # standard loss types
        if 'CrossEntropyLoss' in loss_names:
            ce_kwargs = self.loss_kwargs.get('ce')
            loss_dict['loss_cls'] = cross_entropy_loss(
                logits,
                gt_labels,
                ce_kwargs.get('eps'),
                ce_kwargs.get('alpha')
            ) * ce_kwargs.get('scale')

        if 'TripletLoss' in loss_names:
            tri_kwargs = self.loss_kwargs.get('tri')
            loss_dict['loss_triplet'] = triplet_loss(
                mean_vector,
                gt_labels,
                tri_kwargs.get('margin'),
                tri_kwargs.get('norm_feat'),
                tri_kwargs.get('hard_mining'),
                tri_kwargs.get('metric'),
                variance_vector
            ) * tri_kwargs.get('scale')

        if 'CircleLoss' in loss_names:
            circle_kwargs = self.loss_kwargs.get('circle')
            loss_dict['loss_circle'] = pairwise_circleloss(
                mean_vector,
                gt_labels,
                circle_kwargs.get('margin'),
                circle_kwargs.get('gamma')
            ) * circle_kwargs.get('scale')

        if 'Cosface' in loss_names:
            cosface_kwargs = self.loss_kwargs.get('cosface')
            loss_dict['loss_cosface'] = pairwise_cosface(
                mean_vector,
                gt_labels,
                cosface_kwargs.get('margin'),
                cosface_kwargs.get('gamma'),
            ) * cosface_kwargs.get('scale')

        if 'CenterLoss' in loss_names:
            center_kwargs = self.loss_kwargs.get('center')
            loss_dict['loss_center'] = self.center_loss(mean_vector, gt_labels) * center_kwargs.get('scale')

        # uncertainty specific loss types
        if 'SampleCrossEntropyLoss' in loss_names:
            assert sample_logits_present, "No sample_logits for use in SampleCrossEntropyLoss provided."
            sce_kwargs = self.loss_kwargs.get('sce')
            loss_dict['loss_cls_sample'] = cross_entropy_loss(
                sample_logits,
                gt_labels,
                sce_kwargs.get('eps'),
                sce_kwargs.get('alpha')
            ) * sce_kwargs.get('scale')

        if 'FeatureUncertaintyLoss' in loss_names:
            fu_kwargs = self.loss_kwargs.get('fu')
            loss_dict['loss_fu'] = feature_uncertainty_loss(
                variance_vector,
                fu_kwargs.get('feat_dim')
            ) * fu_kwargs.get('scale')

        if 'DataUncertaintyLoss' in loss_names:
            dul_kwargs = self.loss_kwargs.get('dul')
            loss_dict['loss_dul'] = data_uncertainty_loss(
                logits,
                gt_labels,
                variance_vector,
                dul_kwargs.get('eps'),
                dul_kwargs.get('alpha'),
                dul_kwargs.get('lambda')) * dul_kwargs.get('scale')

        # TODO: what about Bayesian Triplet Loss?

        return loss_dict