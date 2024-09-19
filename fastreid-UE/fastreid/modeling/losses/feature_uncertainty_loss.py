# encoding: utf-8
import torch
import torch.nn.functional as F
import numpy as np

"""
@author:    Andreas Gebhardt
@contact:   AGebhardt1999@gmail.com
"""


ENTROPY_THRESHOLD = np.log(5) + (1+ np.log(2*np.pi))/2 # TODO: maybe add option to deal with multiple samples and make num_samples for DNet configurable

def feature_uncertainty_loss(batched_variance_vector, feat_dim, use_paper_formula=False):

    ENTROPY_CONSTANT = feat_dim*(np.log(2*np.pi) + 1)/2 

    if use_paper_formula:
        # batched_var_vec (B, C)
        sum_of_entropies = 0.5*torch.sum(torch.log(batched_variance_vector + 1e-20)) + batched_variance_vector.shape[0] * ENTROPY_CONSTANT  # add epsilon=1e-20 to avoid infinity # after sum_dim1: (B)

        return F.relu(ENTROPY_THRESHOLD - sum_of_entropies)

    else:
        entropy_loss_avg = 0
        for variance_vector in batched_variance_vector: # iterate over batch
            # softplus scale is used in TF
            entropy = 0.5*torch.sum(torch.log(variance_vector + 1e-20)) + ENTROPY_CONSTANT  # add epsilon=1e-20 to avoid infinity
            entropy_loss_avg += F.relu(ENTROPY_THRESHOLD - entropy/feat_dim) # division by feat_dim=2048 is present in TF code but not explained in paper

        entropy_loss_avg = entropy_loss_avg / batched_variance_vector.shape[0] # average over batch

        return entropy_loss_avg