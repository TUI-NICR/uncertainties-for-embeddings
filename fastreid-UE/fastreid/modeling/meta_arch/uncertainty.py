# encoding: utf-8
"""
@author:    Andreas Gebhardt
@contact:   AGebhardt1999@gmail.com
"""
""" --------------------------------------------------------------------------- BA_UE
This file is based on meta_arch/baseline.py but has been modified for this project.
"""

import itertools
import torch
from torch import nn

from fastreid.config import configurable
from fastreid.modeling.backbones import build_backbone
from fastreid.modeling.crowns import build_crown
from fastreid.modeling.crowns.build import get_crown_feat_dim
from fastreid.modeling.heads import build_mean_head, build_var_head
from fastreid.modeling.losses import *
from fastreid.modeling.losses import mutual_likelihood_score_loss
from fastreid.modeling.necks import build_neck
from fastreid.utils.BA_UE_utils import getattr_rec

from .build import META_ARCH_REGISTRY


@META_ARCH_REGISTRY.register()
class Uncertainty(nn.Module):
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
            parallel_dim_len_eval,
            parallel_dim_repeat,
            parallel_dim_repeat_eval,
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
        self.parallel_dim_len_eval = parallel_dim_len_eval # TODO: should really rename this option and variable names. it is precisely NOT parallel (-> sequential?)
        self.parallel_dim_repeat = parallel_dim_repeat
        self.parallel_dim_repeat_eval = parallel_dim_repeat_eval

        self.modules_to_freeze = modules_to_freeze
        
        # set up components
        self.backbone = backbone
        self.neck = neck
        self.mean_head = mean_head
        self.var_head = var_head
        self.crown = crown

        
        # set up center loss
        if 'CenterLoss' in loss_kwargs['loss_names']:
            self.center_loss = CenterLoss(num_classes=loss_kwargs["center"]["num_classes"], feat_dim=loss_kwargs["center"]["feat_dim"])
        if 'BayesianTripletLoss' in loss_kwargs['loss_names']:
            self.bayesian_triple_loss = BayesianTripletLoss(margin=torch.tensor(0.3)) # TODO: get from config (or better: make into functional call)

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


        self.loss_kwargs = loss_kwargs

        self.register_buffer('pixel_mean', torch.Tensor(pixel_mean).view(1, -1, 1, 1), False)
        self.register_buffer('pixel_std', torch.Tensor(pixel_std).view(1, -1, 1, 1), False)

    @classmethod
    def from_config(cls, cfg):

        modules_to_freeze = cfg.MODEL.FREEZE_WEIGHTS
        enable_3d = cfg.MODEL.HEADS.PARALLEL_3D_CONV
        parallel_dim_len = cfg.MODEL.HEADS.PARALLEL_DIM_LEN
        parallel_dim_len_eval = cfg.MODEL.HEADS.PARALLEL_DIM_LEN_EVAL
        parallel_dim_repeat = cfg.MODEL.HEADS.PARALLEL_DIM_REPEAT
        parallel_dim_repeat_eval = cfg.MODEL.HEADS.PARALLEL_DIM_REPEAT_EVAL

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
            'parallel_dim_len_eval': parallel_dim_len_eval,
            'parallel_dim_repeat': parallel_dim_repeat,
            'parallel_dim_repeat_eval': parallel_dim_repeat_eval,
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
                        'feat_dim': cfg.MODEL.BACKBONE.FEAT_DIM,
                        'use_paper_formula': cfg.MODEL.LOSSES.FU.USE_PAPER_FORMULA
                    },
                    'dul': {
                        'eps': cfg.MODEL.LOSSES.DUL.EPSILON,
                        'lambda': cfg.MODEL.LOSSES.DUL.LAMBDA,
                        'alpha': cfg.MODEL.LOSSES.DUL.ALPHA,
                        'scale': cfg.MODEL.LOSSES.DUL.SCALE
                    },
                    'sdul': {
                        'eps': cfg.MODEL.LOSSES.SDUL.EPSILON,
                        'lambda': cfg.MODEL.LOSSES.SDUL.LAMBDA,
                        'alpha': cfg.MODEL.LOSSES.SDUL.ALPHA,
                        'scale': cfg.MODEL.LOSSES.SDUL.SCALE
                    },
                    'mls': {
                        'scale': cfg.MODEL.LOSSES.MLS.SCALE,
                        'use_raw_mean': cfg.MODEL.LOSSES.MLS.USE_RAW_MEAN
                    }
                }
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        
        #print("batched_inputs has nan", torch.isnan(batched_inputs["images"]).any())

        # get batch of input images
        images = self.preprocess_image(batched_inputs)
        
        # backbone feature extraction
        #print("images has nan", torch.isnan(images).any())
        deep_features = self.backbone(images)

        if self.enable_3d:
            # better than repeating the pipeline (see below) is parallel execution but this costs memory so we have both options (for use with BNN)
            if self.training:
                dim_3d = self.parallel_dim_len
            else:
                dim_3d = self.parallel_dim_len_eval

            # repeat along new dimension to compute with multiple weight samples of neck in parallel
            deep_features = deep_features.unsqueeze(2).expand(-1, -1, dim_3d, -1, -1) 

        # we might repeat the pipeline after this point (for use with BNN)
        if self.training:
            parallel_repeat = self.parallel_dim_repeat
        else: 
            parallel_repeat = self.parallel_dim_repeat_eval

        crown_feat_list = [] # need to aggregate results from the repeats

        for _ in range(parallel_repeat): 

            # preprocessing of deep features and option for BNN
            features_neck = self.neck(deep_features)
            #print("deep features has nan:", torch.isnan(deep_features).any())
            #print("neck feats has nan:", torch.isnan(features_neck).any())

            # calculate parameters for probabilistic embedding ( N(mu, Sigma) )
            features = self.mean_head(features_neck) # mean vector should be located at features["mean_vector"], raw mean data is at features["mean"] (might be the same)
            uncertainty = self.var_head(features_neck) # variance vector should be located at uncertainty["variance_vector"], raw variance data is at uncertainty["variance"] (might be the same)

            # post-processing, option to work with the distribution (probabilistic embedding)
            crown_feats = self.crown(features, uncertainty) # crown feats should always include logits, mean_vector, and variance_vector
            crown_feat_list.append(crown_feats)

        # if we had multiple executions of the pipeline above, we need to collapse it back to a single set of outputs
        # also we can calculate additional information about variance
        if self.enable_3d or parallel_repeat > 1:

            crown_feats = {}

            for key in crown_feat_list[0]:
                if type(crown_feat_list[0][key]) == list:
                    # for sample_logits just concatenate the lists of samples
                    crown_feats[key] = list(itertools.chain.from_iterable([cf[key] for cf in crown_feat_list]))
                    continue

                elif crown_feat_list[0][key] == None:
                    continue
                
                elif len(crown_feat_list[0][key].shape) == 2:
                    # 2D case: shape (B, C)
                    for feat in crown_feat_list:
                        feat[key] = feat[key].unsqueeze(dim=2) # add new dimension

                elif len(crown_feat_list[0][key].shape) == 3:
                    # 3D case: shape (B,C,D)
                    pass
                else:
                    raise TypeError(f"The shape of the crown_feats element at index 0 '{key}' is '{crown_feat_list[0].shape}' with length {len(crown_feat_list[0].shape)} but only lengths 2 or 3 are expected.")

                # the additional rounds from the for loop are to be interpreted as if we had
                # a bigger parallel_dim_len(_eval)
                key_elements = [cf[key] for cf in crown_feat_list]
                stacked_key_elements = torch.cat(key_elements, dim=2)

                if "todo_variance_of" in key:
                    # the crown might have provided a different set of values than the given mean vectors from which
                    # the variance should be calculated, so we do that here
                    crown_feats[key[5:]] = stacked_key_elements.var(dim=2) 
                else:
                    crown_feats[key] = torch.mean(stacked_key_elements, dim=2) # collapse using mean
                    if ("variance_of_" + key) not in crown_feats:
                        # if variance was already calculated from crown-info, don't overwrite
                        crown_feats["variance_of_" + key] = stacked_key_elements.var(dim=2) # also get variance information
        else:
            # case: 2D features and len(list)=1 -> need no processing, can just take the one entry
            crown_feats = crown_feat_list[0]

        if self.training:
            assert "targets" in batched_inputs, "Person ID annotation are missing in training!"
            targets = batched_inputs["targets"]

            # PreciseBN flag, When do preciseBN on different dataset, the number of classes in new dataset
            # may be larger than that in the original dataset, so the circle/arcface will
            # throw an error. We just set all the targets to 0 to avoid this problem.
            if targets.sum() < 0: targets.zero_()

            losses = self.losses(crown_feats, targets)

            return losses
        
        else:
            outputs = {
                "mean_vector": crown_feats["mean_vector"],
                "variance_vector": crown_feats["variance_vector"]
            }
            if self.enable_3d or parallel_repeat > 1:
                outputs["variance_of_mean_vector"] = crown_feats["variance_of_mean_vector"]
                outputs["variance_of_variance_vector"] = crown_feats["variance_of_variance_vector"]

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

        if "mean_vector_raw" in outputs:
            mean_vector_raw = outputs["mean_vector_raw"]
        else:
            mean_vector_raw = mean_vector

        # we sometimes get these
        try: # TODO: would there be a performance increase if I made this part conditional?
            sample_logits = outputs["sample_logits"]
            sample_logits_present = True
        except KeyError:
            sample_logits_present = False

        if logits != None:
            # Log prediction accuracy
            log_accuracy(logits.clone().mul(1).detach(), gt_labels) # TODO: which of these .calls do we need? any?


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
        if 'BayesianTripletLoss' in loss_names:
            btri_kwargs = self.loss_kwargs.get('tri')
            loss_dict['loss_bayesian_triplet'] = self.bayesian_triple_loss(
                mean_vector, # TODO: try using raw mean vector
                variance_vector,
                gt_labels
            ) * btri_kwargs.get('scale')

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

            sce_loss = 0
            for sample_logit in sample_logits:
                sce_loss += cross_entropy_loss(
                    sample_logit,
                    gt_labels,
                    sce_kwargs.get('eps'),
                    sce_kwargs.get('alpha')
                )
            sce_loss /= len(sample_logits)

            loss_dict['loss_cls_sample'] = sce_loss * sce_kwargs.get('scale')

        if 'FeatureUncertaintyLoss' in loss_names:
            fu_kwargs = self.loss_kwargs.get('fu')
            loss_dict['loss_fu'] = feature_uncertainty_loss(
                variance_vector,
                fu_kwargs.get('feat_dim'),
                fu_kwargs.get('use_paper_formula')
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
            
        if 'SampleDataUncertaintyLoss' in loss_names:
            assert sample_logits_present, "No sample_logits for use in SampleDataUncertaintyLoss provided."
            sdul_kwargs = self.loss_kwargs.get('sdul') 

            sdul_loss = 0
            for sample_logit in sample_logits:
                
                sdul_loss += data_uncertainty_loss(
                    sample_logit,
                    gt_labels,
                    variance_vector,
                    sdul_kwargs.get('eps'),
                    sdul_kwargs.get('alpha'),
                    sdul_kwargs.get('lambda')
                )
            sdul_loss /= len(sample_logits)            

            loss_dict['loss_sdul'] = sdul_loss * sdul_kwargs.get('scale')
            
        if 'MutualLikelihoodScoreLoss' in loss_names:
            mls_kwargs = self.loss_kwargs.get('mls')
            if mls_kwargs.get('use_raw_mean'):
                loss_dict['loss_mls'] = mutual_likelihood_score_loss(
                    mean_vector_raw,
                    variance_vector,
                    gt_labels
                ) * mls_kwargs.get('scale')
            else:
                loss_dict['loss_mls'] = mutual_likelihood_score_loss(
                    mean_vector,
                    variance_vector,
                    gt_labels
                ) * mls_kwargs.get('scale')

        # TODO: what about Bayesian Triplet Loss?

        return loss_dict