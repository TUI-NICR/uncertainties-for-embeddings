import torch

"""
@author:    Andreas Gebhardt
@contact:   AGebhardt1999@gmail.com
"""

def negative_MLS(features, features_variance, others, others_variance):
    """
    This function computes the *negative* MLS between two batches of multivariate normal distributions with diagonal covariances.
    """
    # shapes: (B,C)

    # reshape
    features = features.unsqueeze(1) # (M, 1, C)
    features_variance = features_variance.unsqueeze(1)
    others = others.unsqueeze(0) # (1, N, C)
    others_variance = others_variance.unsqueeze(0)
    
    combined_variance = features_variance + others_variance # (M,N,C)

    diffs = torch.square(features - others) / (1e-10 + combined_variance) + torch.log(combined_variance) # const is omitted, probably because it vanishes in gradient
    # there should be a *0.5 here, according to the paper, but it is not present in the code so we omit it here as well
    return torch.sum(diffs, dim=2)


def mutual_likelihood_score_loss(mean_vectors, variance_vectors, labels):
    # labels: Tensor of shape (B), entries are class IDs (e.g. 12, 235, ...)
    batch_size = mean_vectors.shape[0]

    diag_mask = torch.eye(batch_size, dtype=torch.bool, device=mean_vectors.device)
    non_diag_mask = ~diag_mask

    loss_mat = negative_MLS(mean_vectors, variance_vectors, mean_vectors, variance_vectors)
    
    label_mat = labels[:, None] == labels[None, :]
    label_mask_pos = non_diag_mask & label_mat

    loss_pos = loss_mat[label_mask_pos]
    
    return torch.mean(loss_pos)
 