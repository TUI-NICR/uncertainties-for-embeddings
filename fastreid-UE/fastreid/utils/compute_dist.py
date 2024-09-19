# encoding: utf-8
"""
@author:  xingyu liao
@contact: sherlockliao01@gmail.com
"""

# Modified from: https://github.com/open-mmlab/OpenUnReID/blob/66bb2ae0b00575b80fbe8915f4d4f4739cc21206/openunreid/core/utils/compute_dist.py

"""
Modified by Andreas Gebhardt in 2024.
"""

import faiss
import numpy as np
import torch
import torch.nn.functional as F
from datetime import datetime, timedelta
import logging
logger = logging.getLogger(__name__)

from .faiss_utils import (
    index_init_cpu,
    index_init_gpu,
    search_index_pytorch,
    search_raw_array_pytorch,
)

__all__ = [
    "build_dist",
    "compute_jaccard_distance",
    "compute_euclidean_distance",
    "compute_cosine_distance",
]


@torch.no_grad()
def build_dist(feat_1: torch.Tensor, feat_2: torch.Tensor, metric: str = "euclidean", **kwargs) -> np.ndarray:
    r"""Compute distance between two feature embeddings.

    Args:
        feat_1 (torch.Tensor): 2-D feature with batch dimension.
        feat_2 (torch.Tensor): 2-D feature with batch dimension.
        metric:

    Returns:
        numpy.ndarray: distance matrix.
    """

    logger.info(f"computing {metric}...")

    if metric == "euclidean":
        return compute_euclidean_distance(feat_1, feat_2)

    elif metric == "cosine":
        return compute_cosine_distance(feat_1, feat_2)

    elif metric == "jaccard":
        feat = torch.cat((feat_1, feat_2), dim=0)
        dist = compute_jaccard_distance(feat, k1=kwargs["k1"], k2=kwargs["k2"], search_option=0)
        return dist[: feat_1.size(0), feat_1.size(0):]
    
    elif metric == "euclidean_dnet":
        return compute_euclidean_distance_dnet(feat_1, feat_2)
    
    # uncertainty aware metrics
    elif metric == "kl_divergence":
        return compute_kl_divergence(feat_1, kwargs["query_variances"], feat_2, kwargs["gallery_variances"])
    
    elif metric == "js_divergence":
        return compute_js_divergence(feat_1, kwargs["query_variances"], feat_2, kwargs["gallery_variances"])
    
    elif metric == "bhat_distance":
        return compute_bhatt_distance(feat_1, kwargs["query_variances"], feat_2, kwargs["gallery_variances"])
    
    elif metric == "wasserstein":
        return compute_wasserstein_distance(feat_1, kwargs["query_variances"], feat_2, kwargs["gallery_variances"])

    elif metric == "sigma_euclidean":
        return compute_sigma_euclidean_distance(feat_1, kwargs["query_variances"], feat_2, kwargs["gallery_variances"])

    elif metric == "sqrt_sigma_euclidean":
        return compute_sqrt_sigma_euclidean_distance(feat_1, kwargs["query_variances"], feat_2, kwargs["gallery_variances"])

    elif metric == "sigma_cosine":
        return compute_sigma_cosine_distance(feat_1, kwargs["query_variances"], feat_2, kwargs["gallery_variances"])

    elif metric == "sqrt_sigma_cosine":
        return compute_sqrt_sigma_cosine_distance(feat_1, kwargs["query_variances"], feat_2, kwargs["gallery_variances"])

    else:
        raise ValueError(f"Invalid distance measure '{metric}'. You set cfg.TEST.METRIC incorrectly. You can specify a single string or a list of strings (no quotes) but {metric} is not valid. See compute_dist.py for a list of valid options.")


def k_reciprocal_neigh(initial_rank, i, k1):
    forward_k_neigh_index = initial_rank[i, : k1 + 1]
    backward_k_neigh_index = initial_rank[forward_k_neigh_index, : k1 + 1]
    fi = np.where(backward_k_neigh_index == i)[0]
    return forward_k_neigh_index[fi]


@torch.no_grad()
def compute_jaccard_distance(features, k1=20, k2=6, search_option=0, fp16=False):
    if search_option < 3:
        # torch.cuda.empty_cache()
        features = features.cuda()

    ngpus = faiss.get_num_gpus()
    N = features.size(0)
    mat_type = np.float16 if fp16 else np.float32

    if search_option == 0:
        # GPU + PyTorch CUDA Tensors (1)
        res = faiss.StandardGpuResources()
        res.setDefaultNullStreamAllDevices()
        _, initial_rank = search_raw_array_pytorch(res, features, features, k1)
        initial_rank = initial_rank.cpu().numpy()
    elif search_option == 1:
        # GPU + PyTorch CUDA Tensors (2)
        res = faiss.StandardGpuResources()
        index = faiss.GpuIndexFlatL2(res, features.size(-1))
        index.add(features.cpu().numpy())
        _, initial_rank = search_index_pytorch(index, features, k1)
        res.syncDefaultStreamCurrentDevice()
        initial_rank = initial_rank.cpu().numpy()
    elif search_option == 2:
        # GPU
        index = index_init_gpu(ngpus, features.size(-1))
        index.add(features.cpu().numpy())
        _, initial_rank = index.search(features.cpu().numpy(), k1)
    else:
        # CPU
        index = index_init_cpu(features.size(-1))
        index.add(features.cpu().numpy())
        _, initial_rank = index.search(features.cpu().numpy(), k1)

    nn_k1 = []
    nn_k1_half = []
    for i in range(N):
        nn_k1.append(k_reciprocal_neigh(initial_rank, i, k1))
        nn_k1_half.append(k_reciprocal_neigh(initial_rank, i, int(np.around(k1 / 2))))

    V = np.zeros((N, N), dtype=mat_type)
    for i in range(N):
        k_reciprocal_index = nn_k1[i]
        k_reciprocal_expansion_index = k_reciprocal_index
        for candidate in k_reciprocal_index:
            candidate_k_reciprocal_index = nn_k1_half[candidate]
            if len(
                    np.intersect1d(candidate_k_reciprocal_index, k_reciprocal_index)
            ) > 2 / 3 * len(candidate_k_reciprocal_index):
                k_reciprocal_expansion_index = np.append(
                    k_reciprocal_expansion_index, candidate_k_reciprocal_index
                )

        k_reciprocal_expansion_index = np.unique(
            k_reciprocal_expansion_index
        )  # element-wise unique

        x = features[i].unsqueeze(0).contiguous()
        y = features[k_reciprocal_expansion_index]
        m, n = x.size(0), y.size(0)
        dist = (
                torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n)
                + torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        )
        dist.addmm_(x, y.t(), beta=1, alpha=-2)

        if fp16:
            V[i, k_reciprocal_expansion_index] = (
                F.softmax(-dist, dim=1).view(-1).cpu().numpy().astype(mat_type)
            )
        else:
            V[i, k_reciprocal_expansion_index] = (
                F.softmax(-dist, dim=1).view(-1).cpu().numpy()
            )

    del nn_k1, nn_k1_half, x, y
    features = features.cpu()

    if k2 != 1:
        V_qe = np.zeros_like(V, dtype=mat_type)
        for i in range(N):
            V_qe[i, :] = np.mean(V[initial_rank[i, :k2], :], axis=0)
        V = V_qe
        del V_qe

    del initial_rank

    invIndex = []
    for i in range(N):
        invIndex.append(np.where(V[:, i] != 0)[0])  # len(invIndex)=all_num

    jaccard_dist = np.zeros((N, N), dtype=mat_type)
    for i in range(N):
        temp_min = np.zeros((1, N), dtype=mat_type)
        indNonZero = np.where(V[i, :] != 0)[0]
        indImages = [invIndex[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            temp_min[0, indImages[j]] = temp_min[0, indImages[j]] + np.minimum(
                V[i, indNonZero[j]], V[indImages[j], indNonZero[j]]
            )

        jaccard_dist[i] = 1 - temp_min / (2 - temp_min)

    del invIndex, V

    pos_bool = jaccard_dist < 0
    jaccard_dist[pos_bool] = 0.0

    return jaccard_dist


@torch.no_grad()
def compute_euclidean_distance(features, others):
    m, n = features.size(0), others.size(0)
    dist_m = (
            torch.pow(features, 2).sum(dim=1, keepdim=True).expand(m, n)
            + torch.pow(others, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    )
    dist_m.addmm_(features, others.t(), beta=1, alpha=-2) # adjusted to new signature

    return dist_m.cpu().numpy()

@torch.no_grad()
def compute_euclidean_distance_dnet(features, others):
    # Calculate L2 norm for features and others
    features_norm = torch.norm(features, dim=1, keepdim=True)
    others_norm = torch.norm(others, dim=1, keepdim=True)
    
    # Divide each feature vector by its L2 norm
    features = features / features_norm
    others = others / others_norm

    m, n = features.size(0), others.size(0)
    dist_m = (
            torch.pow(features, 2).sum(dim=1, keepdim=True).expand(m, n)
            + torch.pow(others, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    )
    dist_m.addmm_(1, -2, features, others.t())

    return dist_m.cpu().numpy()


@torch.no_grad()
def compute_cosine_distance(features, others):
    """Computes cosine distance.
    Args:
        features (torch.Tensor): 2-D feature matrix.
        others (torch.Tensor): 2-D feature matrix.
    Returns:
        torch.Tensor: distance matrix.
    """
    features = F.normalize(features, p=2, dim=1)
    others = F.normalize(others, p=2, dim=1)
    dist_m = 1 - torch.mm(features, others.t())
    return dist_m.cpu().numpy()


# We need a distance matrix where dist[i][j] = d(features[i], others[j])

# TODO: maybe: remove timing again? does it impact performance?

@torch.no_grad()
def compute_sigma_euclidean_distance(features, features_variance, others, others_variance):
    """
    Computes a weighted euclidean distance: d_euclid(features / features_variance, otheres / others_variance).
    
    Args:
        features (torch.Tensor): (m,d) matrix of feature means
        features_variance (torch.Tensor): (m,d) matrix of features covariance matrix diagonals
        others (torch.Tensor): (n,d) matrix of other means
        others_variance (torch.Tensor): (n,d) matrix of others covariance matrix diagonals
    Returns:
        torch.Tensor: (m,n) distance matrix, dist[i][j] = d(features[i], others[j])
    """

    return compute_euclidean_distance(features / features_variance, others / others_variance)


@torch.no_grad()
def compute_sqrt_sigma_euclidean_distance(features, features_variance, others, others_variance):
    """
    Computes a weighted euclidean distance: d_euclid(features / sqrt(features_variance), otheres / sqrt(others_variance)).
    
    Args:
        features (torch.Tensor): (m,d) matrix of feature means
        features_variance (torch.Tensor): (m,d) matrix of features covariance matrix diagonals
        others (torch.Tensor): (n,d) matrix of other means
        others_variance (torch.Tensor): (n,d) matrix of others covariance matrix diagonals
    Returns:
        torch.Tensor: (m,n) distance matrix, dist[i][j] = d(features[i], others[j])
    """

    return compute_euclidean_distance(features / torch.sqrt(features_variance), 
                                      others / torch.sqrt(others_variance))

@torch.no_grad()
def compute_sigma_cosine_distance(features, features_variance, others, others_variance):
    """
    Computes a weighted cosine distance: d_cos(features / features_variance, otheres / others_variance).
    
    Args:
        features (torch.Tensor): (m,d) matrix of feature means
        features_variance (torch.Tensor): (m,d) matrix of features covariance matrix diagonals
        others (torch.Tensor): (n,d) matrix of other means
        others_variance (torch.Tensor): (n,d) matrix of others covariance matrix diagonals
    Returns:
        torch.Tensor: (m,n) distance matrix, dist[i][j] = d(features[i], others[j])
    """

    return compute_cosine_distance(features / features_variance, others / others_variance)


@torch.no_grad()
def compute_sqrt_sigma_cosine_distance(features, features_variance, others, others_variance):
    """
    Computes a weighted cosine distance: d_cos(features / sqrt(features_variance), otheres / sqrt(others_variance)).
    
    Args:
        features (torch.Tensor): (m,d) matrix of feature means
        features_variance (torch.Tensor): (m,d) matrix of features covariance matrix diagonals
        others (torch.Tensor): (n,d) matrix of other means
        others_variance (torch.Tensor): (n,d) matrix of others covariance matrix diagonals
    Returns:
        torch.Tensor: (m,n) distance matrix, dist[i][j] = d(features[i], others[j])
    """

    return compute_cosine_distance(features / torch.sqrt(features_variance), 
                                      others / torch.sqrt(others_variance))



@torch.no_grad()
def _blockwise_divergence_calculation(func, features, features_variance, others, others_variance,
                           features_block_size=512, others_block_size=1024):
    """
    Applies func to combinations of sections of features and others to fill out the resulting dist matrix in blocks.
    This is to reduce the RAM requirement during calculation.

    Args:
        func (function): function for calculating the divergence
        features (torch.Tensor): (m,d) matrix of feature means
        features_variance (torch.Tensor): (m,d) matrix of features covariance matrix diagonals
        others (torch.Tensor): (n,d) matrix of other means
        others_variance (torch.Tensor): (n,d) matrix of others covariance matrix diagonals
    Returns:
        torch.Tensor: (m,n) distance matrix, dist[i][j] = d(features[i], others[j])
    """
    start = datetime.now()
    m, n = features.size(0), others.size(0)

    dist = torch.zeros(m, n, device=features.device)

    finished = False
    feat_start = 0

    block_times = []
    incomplete_block_times = []

    while not finished:

        feat_end = feat_start + features_block_size
        
        if feat_end >= m: 
            feat_end = m
            finished = True

        row_finished = False
        oth_start = 0

        while not row_finished:
            block_start = datetime.now()
            
            oth_end = oth_start + others_block_size

            if oth_end >= n: 
                oth_end = n
                row_finished = True

            dist_block = func(features[feat_start:feat_end], 
                                features_variance[feat_start:feat_end], 
                                others[oth_start:oth_end], 
                                others_variance[oth_start:oth_end])
            
            dist[feat_start:feat_end, oth_start:oth_end] = dist_block

            oth_start = oth_end # next iteration starts there
            if not (finished or row_finished):
                block_times.append(datetime.now()-block_start)
            else:
                incomplete_block_times.append(datetime.now()-block_start)

        feat_start = feat_end # next iterations starts there

    end = datetime.now()
    logger.info(f"Blockwise compute with '{func.__name__}' in {len(block_times) + len(incomplete_block_times)} blocks took {end-start}")
    logger.info(f"{len(block_times)} complete blocks took on average {sum(block_times, timedelta())/len(block_times)}")
    logger.info(f"{len(incomplete_block_times)} incomplete blocks took on average {sum(incomplete_block_times, timedelta())/len(incomplete_block_times)}")
    return dist

@torch.no_grad()
def compute_kl_divergence(features, features_variance, others, others_variance):
    """
    Computes the KL-divergence between N(features, features_variance @ I) and N(others, others_variance @ I).
    `N` is normal distribution, `@ I` is matrix multiplication with identity matrix.

    This functions splits the computation into blocks because otherwise the RAM requirements are too great.
    
    Args:
        features (torch.Tensor): (m,d) matrix of feature means
        features_variance (torch.Tensor): (m,d) matrix of features covariance matrix diagonals
        others (torch.Tensor): (n,d) matrix of other means
        others_variance (torch.Tensor): (n,d) matrix of others covariance matrix diagonals
    Returns:
        torch.Tensor: (m,n) distance matrix, dist[i][j] = d(features[i], others[j])
    """
    
    return _blockwise_divergence_calculation(
        _compute_kl_divergence, features, features_variance, others, others_variance).cpu().numpy()
    
@torch.no_grad()
def compute_js_divergence(features, features_variance, others, others_variance):
    """
    Computes the JS-divergence between N(features, features_variance @ I) and N(others, others_variance @ I).
    `N` is normal distribution, `@ I` is matrix multiplication with identity matrix.

    This functions splits the computation into blocks because otherwise the RAM requirements are too great.
    
    Args:
        features (torch.Tensor): (m,d) matrix of feature means
        features_variance (torch.Tensor): (m,d) matrix of features covariance matrix diagonals
        others (torch.Tensor): (n,d) matrix of other means
        others_variance (torch.Tensor): (n,d) matrix of others covariance matrix diagonals
    Returns:
        torch.Tensor: (m,n) distance matrix, dist[i][j] = d(features[i], others[j])
    """
    
    return _blockwise_divergence_calculation(
            _compute_js_divergence, 
            features, 
            features_variance, 
            others, 
            others_variance, 
            256, 
            512
        ).cpu().numpy()

@torch.no_grad()
def compute_bhatt_distance(features, features_variance, others, others_variance):
    """
    Computes the Bhattacharyya distance between N(features, features_variance @ I) and N(others, others_variance @ I).
    `N` is normal distribution, `@ I` is matrix multiplication with identity matrix.

    This functions splits the computation into blocks because otherwise the RAM requirements are too great.
    
    Args:
        features (torch.Tensor): (m,d) matrix of feature means
        features_variance (torch.Tensor): (m,d) matrix of features covariance matrix diagonals
        others (torch.Tensor): (n,d) matrix of other means
        others_variance (torch.Tensor): (n,d) matrix of others covariance matrix diagonals
    Returns:
        torch.Tensor: (m,n) distance matrix, dist[i][j] = d(features[i], others[j])
    """
    
    return _blockwise_divergence_calculation(
        _compute_bhatt_distance, features, features_variance, others, others_variance).cpu().numpy()

@torch.no_grad()
def compute_wasserstein_distance(features, features_variance, others, others_variance):
    """
    Computes the (squared) 2-Wasserstein distance W_2^2 between N(features, features_variance @ I) and N(others, others_variance @ I).
    `N` is normal distribution, `@ I` is matrix multiplication with identity matrix.

    This functions splits the computation into blocks because otherwise the RAM requirements are too great.
    
    Args:
        features (torch.Tensor): (m,d) matrix of feature means
        features_variance (torch.Tensor): (m,d) matrix of features covariance matrix diagonals
        others (torch.Tensor): (n,d) matrix of other means
        others_variance (torch.Tensor): (n,d) matrix of others covariance matrix diagonals
    Returns:
        torch.Tensor: (m,n) distance matrix, dist[i][j] = d(features[i], others[j])
    """
    
    return _blockwise_divergence_calculation(
        _compute_wasserstein_distance, features, features_variance, others, others_variance).cpu().numpy()

@torch.no_grad()
def _compute_kl_divergence(features, features_variance, others, others_variance) -> torch.Tensor:
    """
    Computes the KL-divergence between N(features, features_variance @ I) and N(others, others_variance @ I).
    `N` is normal distribution, `@ I` is matrix multiplication with identity matrix.
    
    Args:
        features (torch.Tensor): (m,d) matrix of feature means
        features_variance (torch.Tensor): (m,d) matrix of features covariance matrix diagonals
        others (torch.Tensor): (n,d) matrix of other means
        others_variance (torch.Tensor): (n,d) matrix of others covariance matrix diagonals
    Returns:
        torch.Tensor: (m,n) distance matrix, dist[i][j] = d(features[i], others[j])
    """

    # Reshape the tensors for broadcasting
    features = features.unsqueeze(1)  # (m, 1, d)
    features_variance = features_variance.unsqueeze(1)  # (m, 1, d)
    others = others.unsqueeze(0)  # (1, n, d)
    others_variance = others_variance.unsqueeze(0)  # (1, n, d)

    features_variance_det = torch.prod(features_variance, dim=2) + 1e-10  # (m, 1) # prone to becoming 0 because 0.5**2048 -> 0 so we add eps>0
    others_variance_det = torch.prod(others_variance, dim=2) + 1e-10  # (1, n)
    inv_others_variance = 1. / others_variance  # (1, n, d)

    diff = others - features  # (m, n, d)

    kl_divergence = 0.5 * (                                          # features = p, others = q
        torch.log(others_variance_det / features_variance_det)       # log |\Sigma_q| / |\Sigma_p|                   # (m, n) # if both are 1e-10, log(1)=0
        + torch.sum(inv_others_variance * features_variance, dim=2)  # + tr(\Sigma_q^{-1} * \Sigma_p)                # (m, n)
        + torch.sum(diff * inv_others_variance * diff, dim=2)        # + (\mu_q-\mu_p)^T\Sigma_q^{-1}(\mu_q-\mu_p)   # (m, n)
        - features.size(2)                                           # - N                                           # scalar
    )

    return kl_divergence    # tested and should work

@torch.no_grad()
def _compute_bhatt_distance(features, features_variance, others, others_variance):
    """
    Computes the Bhattacharyya distance between N(features, features_variance @ I) and N(others, others_variance @ I).
    `N` is normal distribution, `@ I` is matrix multiplication with identity matrix.
    
    Args:
        features (torch.Tensor): (m,d) matrix of feature means
        features_variance (torch.Tensor): (m,d) matrix of features covariance matrix diagonals
        others (torch.Tensor): (n,d) matrix of other means
        others_variance (torch.Tensor): (n,d) matrix of others covariance matrix diagonals
    Returns:
        torch.Tensor: (m,n) distance matrix, dist[i][j] = d(features[i], others[j])
    """

    diff = others.unsqueeze(0) - features.unsqueeze(1) # (m, n, d)
    
    avg_var = (features_variance.unsqueeze(1) + others_variance.unsqueeze(0)) / 2.0 # (m, n, d)

    # Log-determinants of variances
    log_det_features_var = torch.log(torch.prod(features_variance, dim=1, keepdim=True) + 1e-10) # (m, 1)
    log_det_others_var = torch.log(torch.prod(others_variance, dim=1) + 1e-10).unsqueeze(0) # (1, n)
    log_det_avg_var = torch.log(torch.prod(avg_var, dim=2) + 1e-10) # (m, n)

    # 0.5 log(|\Sigma_{pq}| / sqrt(\Sigma_p * \Sigma_q))
    norm = 0.5 * (log_det_avg_var - 0.5 * (log_det_features_var + log_det_others_var)) # (m, n)

    # 0.125 (\mu_q - \mu_p)^T \Sigma_{pq}^{-1} (\mu_q - \mu_p)
    dist = 0.125 * (diff * (1.0 / avg_var) * diff).sum(dim=2) # (m, n)

    return dist + norm    # tested and should work

@torch.no_grad()
def _compute_js_divergence(features, features_variance, others, others_variance):
    """
    Computes the Jenson-Shannon divergence between N(features, features_variance @ I) and N(others, others_variance @ I).
    `N` is normal distribution, `@ I` is matrix multiplication with identity matrix.
    
    Args:
        features (torch.Tensor): (m,d) matrix of feature means
        features_variance (torch.Tensor): (m,d) matrix of features covariance matrix diagonals
        others (torch.Tensor): (n,d) matrix of other means
        others_variance (torch.Tensor): (n,d) matrix of others covariance matrix diagonals
    Returns:
        torch.Tensor: (m,n) distance matrix, dist[i][j] = d(features[i], others[j])
    """

    # Reshape the tensors for broadcasting
    features = features.unsqueeze(1)  # (m, 1, d)
    features_variance = features_variance.unsqueeze(1)  # (m, 1, d)
    others = others.unsqueeze(0)  # (1, n, d)
    others_variance = others_variance.unsqueeze(0)  # (1, n, d)

    mixture = (features + others) / 2.0 # (m, n, d)
    mixture_variance = (features_variance + others_variance) / 4.0 # (m, n, d)
    
    def _kl(features, features_variance, others, others_variance): # helper function that takes already expanded inputs
        features_variance_det = torch.prod(features_variance, dim=2) + 1e-10  # (m, 1)
        others_variance_det = torch.prod(others_variance, dim=2) + 1e-10  # (1, n)
        inv_others_variance = 1. / others_variance  # (1, n, d)

        diff = others - features  # (m, n, d)

        kl_divergence = 0.5 * (                                          # features = p, others = q
            torch.log(others_variance_det / features_variance_det)       # log |\Sigma_q| / |\Sigma_p|                   # (m, n)
            + torch.sum(inv_others_variance * features_variance, dim=2)  # + tr(\Sigma_q^{-1} * \Sigma_p)                # (m, n)
            + torch.sum(diff * inv_others_variance * diff, dim=2)        # + (\mu_q-\mu_p)^T\Sigma_q^{-1}(\mu_q-\mu_p)   # (m, n)
            - features.size(2)                                           # - N                                           # scalar
        )

        return kl_divergence

    kl1 = _kl(features, features_variance, mixture, mixture_variance)
    kl2 = _kl(others,   others_variance,   mixture, mixture_variance)

    return 0.5 * (kl1 + kl2)

@torch.no_grad()
def _compute_wasserstein_distance(features, features_variance, others, others_variance):
    """
    Computes the (squared) 2-Wasserstein distance W_2^2 between N(features, features_variance @ I) and N(others, others_variance @ I).
    `N` is normal distribution, `@ I` is matrix multiplication with identity matrix.
    
    Args:
        features (torch.Tensor): (m,d) matrix of feature means
        features_variance (torch.Tensor): (m,d) matrix of features covariance matrix diagonals
        others (torch.Tensor): (n,d) matrix of other means
        others_variance (torch.Tensor): (n,d) matrix of others covariance matrix diagonals
    Returns:
        torch.Tensor: (m,n) distance matrix, dist[i][j] = d(features[i], others[j])
    """

    features_variance = features_variance.unsqueeze(1) # (m, 1, d)
    others_variance = others_variance.unsqueeze(0) # (1, n, d)

    m, n = features.size(0), others.size(0)
    dist = (
            torch.pow(features, 2).sum(dim=1, keepdim=True).expand(m, n)
            + torch.pow(others, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    )
    dist.addmm_(features, others.t(), beta=1, alpha=-2) # adjusted to new signature

    # C_2^0.5 @ C_1 @ C_2^0.5 simplifies to C_1 * C_2 because of the diagonal covariances # (bures metric)
    trace = torch.sum(features_variance + others_variance - 2 * torch.sqrt(features_variance * others_variance), dim=2) # (m, n)
    
    return dist + trace
