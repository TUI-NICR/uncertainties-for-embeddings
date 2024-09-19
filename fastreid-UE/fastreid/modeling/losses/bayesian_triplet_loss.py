import torch.nn as nn
import torch

from .utils import cosine_dist, euclidean_dist
from fastreid.utils.compute_dist import _compute_kl_divergence, _compute_js_divergence, _compute_bhatt_distance, _compute_wasserstein_distance

"""
@author:    Andreas Gebhardt
@contact:   AGebhardt1999@gmail.com
"""

"""
Adapted from the code provided in their paper. Not entirely finished.
"""

class BayesianTripletLoss(nn.Module):
    def __init__(self, margin) -> None:
        super().__init__()

        self.margin = margin
        # TODO: should probably move this away from class and to a functional implementation in line with FastReID Triplet Loss

    def forward(self, mean_vectors, variance_vectors, targets):

        # TODO: handle these placeholders
        MINING_TYPE = "proxy"
        metric = "cosine"
        normalization_type = "gauss"
        norm_scale = 1e-6


        # TODO: implement other mining strategies
        # experiments: {brute, random, proxy x {cos, euc, kl, js, bhatt, wass}} x {gauss, vMF, None} -> 24
        # could also do scale hyperparam search. maybe in second step?


        if MINING_TYPE == "brute_force":
            # all possible combinations
            raise NotImplementedError("Bayesian Triplet Loss brute force.")
        elif MINING_TYPE == "random":
            # randomly choose triplets
            raise NotImplementedError("Bayesian Triplet Loss random.")
        
        elif MINING_TYPE == "proxy":
            # use some other metric for mining 
            if metric == "cosine":# or (metric=="euclidean" and norm_feat):
                dist_mat = cosine_dist(mean_vectors, mean_vectors)
            elif metric == "kl_divergence":
                dist_mat = _compute_kl_divergence(mean_vectors, variance_vectors, mean_vectors, variance_vectors)
            elif metric == "js_divergence":
                dist_mat = _compute_js_divergence(mean_vectors, variance_vectors, mean_vectors, variance_vectors)
            elif metric == "bhatt_distance":
                dist_mat = _compute_bhatt_distance(mean_vectors, variance_vectors, mean_vectors, variance_vectors)
            elif metric == "wasserstein":
                dist_mat = _compute_wasserstein_distance(mean_vectors, variance_vectors, mean_vectors, variance_vectors)
            else:
                dist_mat = euclidean_dist(mean_vectors, mean_vectors)

            # For distributed training, gather all features from different process.
            # if comm.get_world_size() > 1:
            #     all_embedding = torch.cat(GatherLayer.apply(embedding), dim=0)
            #     all_targets = concat_all_gather(targets)
            # else:
            #     all_embedding = embedding
            #     all_targets = targets 

            N = dist_mat.size(0)
            is_pos = targets.view(N, 1).expand(N, N).eq(targets.view(N, 1).expand(N, N).t()).float()
            is_neg = targets.view(N, 1).expand(N, N).ne(targets.view(N, 1).expand(N, N).t()).float()

            # using hard mining 
            assert len(dist_mat.size()) == 2
            # copied from triplet_loss.py but here we care about indices, not distances
            _, index_ap = torch.max(dist_mat * is_pos, dim=1)
            _, index_an = torch.min(dist_mat * is_neg + is_pos * 1e9, dim=1)

            # index_ap, index_an: (B)
            # for each x[i] in the batch (A), we get a P as x[index_ap[i]], and an N as x[index_an[i]]. This is our triplet. 

            mean_vectors_P = mean_vectors[index_ap, :]
            variance_vectors_P = variance_vectors[index_ap, :]
            mean_vectors_N = mean_vectors[index_an, :]
            variance_vectors_N = variance_vectors[index_an, :]
                
        else:
            raise ValueError(f"Invalid mining type in Bayesian Triplet Loss. Must be 'brute_force', 'random', or 'proxy' but got '{MINING_TYPE}'.")

        
        btl = self.calculate_BTL(
                        mean_vectors, 
                        mean_vectors_P, 
                        mean_vectors_N, 
                        variance_vectors, 
                        variance_vectors_P, 
                        variance_vectors_N, 
                        self.margin
                    )
        
        # optional additional normalization
        if normalization_type == "gauss":
            mu_prior = torch.zeros_like(mean_vectors[0,:], requires_grad=False).unsqueeze_(0) # we want to compare all mean vectors with the same prior -> "Batch" of size 1 for other
            var_prior = torch.ones_like(variance_vectors[0,:], requires_grad=False).unsqueeze_(0) / mean_vectors.shape[1] # TODO: in the paper this is the value but in the code there is an option to specify this value, although no standard value is given. leaving as this for now, but might want to add config option.

            norm_term = _compute_kl_divergence(mean_vectors, variance_vectors, mu_prior, var_prior).mean() \
                            + _compute_kl_divergence(mean_vectors_P, variance_vectors_P, mu_prior, var_prior).mean() \
                            + _compute_kl_divergence(mean_vectors_N, variance_vectors_N, mu_prior, var_prior).mean() # compute_kl-call returns (B, 1) tensor
        
        elif normalization_type == "vMF":
            # it is unclear what exactly this has to do with the KL divergence between a vMF distribution and ... something unspecified.
            # however, this is equivalent to the term used in the code in the supplemetary material of the original publication.
            norm_term = (1.0 / variance_vectors).mean() + (1.0 / variance_vectors_P).mean() + (1.0 / variance_vectors_N).mean() - 3.0 * mean_vectors.shape[1] * torch.log(2.0)
        else:
            norm_term = 0

        return btl + norm_scale * norm_term
    

    def calculate_BTL(self, mean_A, mean_P, mean_N, variance_A, variance_P, variance_N, margin):
        
        mean_A_sq = mean_A**2
        mean_P_sq = mean_P**2
        mean_N_sq = mean_N**2
        variance_P_sq = variance_P**2
        variance_N_sq = variance_N**2

        mu = torch.sum(mean_P_sq + variance_P - mean_N_sq - variance_N - 2*mean_A*(mean_P - mean_N), dim=1)

        T1 = variance_P_sq + 2*mean_P_sq * variance_P \
                + 2*(variance_A + mean_A_sq)*(variance_P + mean_P_sq) \
                - 2*mean_A_sq * mean_P_sq - 4*mean_A*mean_P*variance_P
        
        T2 = variance_N_sq + 2*mean_N_sq * variance_N \
                + 2*(variance_A + mean_A_sq)*(variance_N + mean_N_sq) \
                - 2*mean_A_sq * mean_N_sq - 4*mean_A*mean_N*variance_N
        
        T3 = 4*mean_P*mean_N*variance_A

        sigma_sq = torch.sum(2*T1 + 2*T2 - 2*T3, dim=1)
        sigma = torch.sqrt(sigma_sq)

        probs = torch.distributions.normal.Normal(loc = mu, scale = sigma + 1e-8).cdf(margin) # it might be preferable to use .clamp(1e-8) instead, however for now it is left as in the original
        nll = -torch.log(probs + 1e-8)

        return nll.mean()