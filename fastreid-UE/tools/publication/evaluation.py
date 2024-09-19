import json
import numpy as np
import torch
import torch.nn.functional as F
from tabulate import tabulate

MODEL_OUTPUTS_PATH = "../trained_model/raw_model_outputs.json"
# MODEL_OUTPUTS_PATH = "../custom_model/raw_model_outputs.json"


MEAN_VEC = 'mean_vector' # feature vector
VAR_OF_MEAN = 'variance_of_mean_vector' # model uncertainty
VAR_VEC = 'variance_vector' # data uncertainty
VAR_OF_VAR = 'variance_of_variance_vector' # distributional uncertainty

def load_data(path=MODEL_OUTPUTS_PATH):
    """loads a raw_model_outputs.json file specified by path"""
    
    print(f"loading {path}...")
    
    with open(path, 'r') as data_file:
        data = json.load(data_file)
    
    print("DONE!")
    
    return data

def get_labels(data, set_id):
    """Returns the person IDs (int) for each element of the given set as an ndarray."""
    # for formatting of the filaname string see documentation:
    # https://zheng-lab.cecs.anu.edu.au/Project/project_reid.html
    # person ID is first four digits of the filename
    return np.array([int(s[:4]) for s in sorted(data['sets'][set_id])])

def get_camera_ids(data, set_id):
    """Returns the camera IDs (int) for each element of the given set as an ndarray."""
    # for formatting of the filaname string see documentation:
    # https://zheng-lab.cecs.anu.edu.au/Project/project_reid.html
    return np.array([int(s[6]) for s in sorted(data['sets'][set_id])])

def precisions(dist_mat, query_labels, gallery_labels,
               query_camera_ids, gallery_camera_ids) -> list[tuple[float, np.ndarray]]:
    """Computes precisions for distance matrix needed to compute mean average
    precision (mAP) or mean precision for each gallery sample.
    Identical mAP results to fastreid.evaluation.rank.eval_market1501 up to
    11th decimal, but much faster computation."""

    cam_mask = np.zeros_like(dist_mat, dtype=bool)
    for i, (q_label, cid) in enumerate(zip(query_labels, query_camera_ids)):
        cam_mask[i] = (gallery_labels == q_label) & (gallery_camera_ids == cid)

    dist_mat_masked = np.array(dist_mat)
    dist_mat_masked[cam_mask] += np.inf

    idxs = np.arange(dist_mat_masked.shape[1], dtype=int)
    sort_idx = np.argsort(dist_mat_masked, axis=1)
    ranks = np.empty_like(dist_mat_masked, dtype=int)
    for i in range(len(dist_mat_masked)):
        ranks[i, sort_idx[i]] = idxs

    precision_and_sorted_matches = []
    for i in range(len(query_labels)):
        matches = idxs[np.logical_and(
            gallery_labels == query_labels[i],
            gallery_camera_ids != query_camera_ids[i])]
        match_ranks = ranks[i][matches]
        match_sort_idx = np.argsort(match_ranks)
        sorted_matches = matches[match_sort_idx]
        sorted_match_ranks = match_ranks[match_sort_idx]
        num_matches = idxs[:len(sorted_match_ranks)] + 1
        num_mismatches = sorted_match_ranks - idxs[:len(match_ranks)]
        precision = num_matches / (num_matches + num_mismatches)
        precision_and_sorted_matches.append((precision, sorted_matches))

    return precision_and_sorted_matches

def map_from_dist_mat(dist_mat, query_labels, gallery_labels,
                      query_camera_ids, gallery_camera_ids):
    """Computes mean average precision for distance matrix.
    Identical results to fastreid.evaluation.rank.eval_market1501 up to
    11th decimal, but much faster computation."""

    prec_list = precisions(dist_mat, query_labels, gallery_labels,
                           query_camera_ids, gallery_camera_ids)

    ap = np.empty_like(query_labels, dtype=float)
    for i, (precision, _) in enumerate(prec_list):
        average_precision = np.mean(precision)
        ap[i] = average_precision
    map_ = np.mean(ap)
    
    return map_

def rank1_from_dist_mat(dist_mat, query_labels, gallery_labels,
                        query_camera_ids, gallery_camera_ids) -> float:
    """Computes the rank-1 accuracy for the given distance matrix."""
    # Initialize the count of correct top-1 matches
    correct_top1_matches = 0
    
    # Loop over each query
    for i in range(len(query_labels)):
        # Mask to ignore the same camera id and label
        cam_mask = (gallery_labels == query_labels[i]) & (gallery_camera_ids == query_camera_ids[i])
        
        # Copy the distance matrix for the current query and apply the mask
        dist_vec = np.array(dist_mat[i])
        dist_vec[cam_mask] = np.inf
        
        # Find the index of the minimum distance (the closest match)
        top1_index = np.argmin(dist_vec)
        
        # Check if the closest match has the same label as the query
        if gallery_labels[top1_index] == query_labels[i]:
            correct_top1_matches += 1
            
    # Compute the rank-1 accuracy
    rank1_accuracy = correct_top1_matches / len(query_labels)
    
    return rank1_accuracy

def get_mAP(dist_mat, data):
    """wrapper for `map_from_dist_mat` to avoid boilerplate"""
    return map_from_dist_mat(
                dist_mat, 
                get_labels(data, "Q"), 
                get_labels(data, "G"),
                get_camera_ids(data, "Q"),
                get_camera_ids(data, "G") )

def get_rank1(dist_mat, data):
    """wrapper for `rank1_from_dist_mat` to avoid boilerplate"""
    return rank1_from_dist_mat(
                dist_mat, 
                get_labels(data, "Q"), 
                get_labels(data, "G"),
                get_camera_ids(data, "Q"),
                get_camera_ids(data, "G") )

def get_vectors(data, set_id, vector_type):
    """Returns the requested raw model output vectors for the set in a list."""
    
    return torch.Tensor([data['data'][name][vector_type]
            for name in sorted(data['sets'][set_id])])

def get_QG_vecs(data, vector_type):
    """wrapper for getting query and gallery vectors to avoid boilerplate"""
    return torch.Tensor(get_vectors(data, "Q", vector_type)), torch.Tensor(get_vectors(data, "G", vector_type))


@torch.no_grad()
def compute_cosine_distance(features, others):
    """Computes cosine distance.

    Args:
        features (torch.Tensor): 2-D feature matrix.
        others (torch.Tensor): 2-D feature matrix.
    Returns:
        ndarray: distance matrix.
    """
    features = F.normalize(features, p=2, dim=1)
    others = F.normalize(others, p=2, dim=1)
    dist_m = 1 - torch.mm(features, others.t())
    return dist_m.cpu().numpy()

def get_vanilla_performance(data):

    print("computing vanilla performance...")

    query_feature_vectors = get_vectors(data, "Q", MEAN_VEC)
    gallery_feature_vectors = get_vectors(data, "G", MEAN_VEC)

    dist_mat = compute_cosine_distance(query_feature_vectors, gallery_feature_vectors)

    mAP = get_mAP(dist_mat, data)
    rank1 = get_rank1(dist_mat, data)

    return 100 * mAP, 100 * rank1

def get_const_c_performance(data, c=0.12249087684867076, augmentation_basis=VAR_OF_MEAN, augmentation_basis_is_std=False):

    print("computing const c performance...")

    query_unc_vecs, gallery_unc_vecs = get_QG_vecs(data, augmentation_basis)
    query_fvs, gallery_fvs = get_QG_vecs(data, MEAN_VEC)

    if augmentation_basis_is_std:
        query_unc_vecs = torch.sqrt(query_unc_vecs)
        gallery_unc_vecs = torch.sqrt(gallery_unc_vecs)

    weighted_query_fvs = query_fvs / (query_unc_vecs + c)
    weighted_gallery_fvs = gallery_fvs / (gallery_unc_vecs + c)

    dist_mat = compute_cosine_distance(weighted_query_fvs, weighted_gallery_fvs)

    mAP = get_mAP(dist_mat, data)
    rank1 = get_rank1(dist_mat, data)

    return 100 * mAP, 100 * rank1

def get_derived_c_performance(data, lambda_=1024, augmentation_basis=VAR_OF_MEAN, augmentation_basis_is_std=False,
                              augmentation_auxiliary=VAR_OF_MEAN, augmentation_auxiliary_is_std=False,
                              score_func=lambda x: 1 / torch.norm(torch.log(x), 1, 1, keepdim=True) ):

    print("computing derived c performance...")

    query_unc_vecs, gallery_unc_vecs = get_QG_vecs(data, augmentation_basis)
    query_aux_unc_vecs, gallery_aux_unc_vecs = get_QG_vecs(data, augmentation_auxiliary)
    query_fvs, gallery_fvs = get_QG_vecs(data, MEAN_VEC)

    if augmentation_basis_is_std:
        query_unc_vecs = torch.sqrt(query_unc_vecs)
        gallery_unc_vecs = torch.sqrt(gallery_unc_vecs)
    if augmentation_auxiliary_is_std:
        query_aux_unc_vecs = torch.sqrt(query_aux_unc_vecs)
        gallery_aux_unc_vecs = torch.sqrt(gallery_aux_unc_vecs)

    query_c = lambda_ * score_func(query_aux_unc_vecs)
    gallery_c = lambda_ * score_func(gallery_aux_unc_vecs)

    weighted_query_fvs = query_fvs / (query_unc_vecs + query_c)
    weighted_gallery_fvs = gallery_fvs / (gallery_unc_vecs + gallery_c)

    dist_mat = compute_cosine_distance(weighted_query_fvs, weighted_gallery_fvs)

    mAP = get_mAP(dist_mat, data)
    rank1 = get_rank1(dist_mat, data)

    return 100 * mAP, 100 * rank1

def get_uncertainty_values(data, index, set_id, score_func):
    mod_unc_Q_vecs, mod_unc_G_vecs = get_QG_vecs(data, VAR_OF_MEAN)
    dat_unc_Q_vecs, dat_unc_G_vecs = get_QG_vecs(data, VAR_VEC)
    dis_unc_Q_vecs, dis_unc_G_vecs = get_QG_vecs(data, VAR_OF_VAR)

    if set_id == "Q":
        return score_func(mod_unc_Q_vecs[index]), score_func(dat_unc_Q_vecs[index]), score_func(dis_unc_Q_vecs[index]), sorted(data["sets"][set_id])[index]
    elif set_id == "G":
        return score_func(mod_unc_G_vecs[index]), score_func(dat_unc_G_vecs[index]), score_func(dis_unc_G_vecs[index]), sorted(data["sets"][set_id])[index]
    else:
        raise ValueError(f"get_uncertainty_values only accepts the set_ids 'Q' and 'G' but '{set_id}' was given.")


if __name__ == "__main__":

    # read raw_model_outputs.json
    data = load_data()

    # get performances with different eval methods
    headers = ["variant", "mAP [%]", "rank-1 [%]"]
    results = [
        ["UAL", *get_vanilla_performance(data)],
        ["UBER (const c, model)", *get_const_c_performance(data)],
        ["UBER (const c, sqrt(model))", *get_const_c_performance(data, augmentation_basis_is_std=True, c=0.24346568542494923)],
        ["UBER (derived c, model/model)", *get_derived_c_performance(data)],
        ["UBER (derived c, model/dist)", *get_derived_c_performance(data, lambda_=0.0469218, augmentation_auxiliary=VAR_OF_VAR, 
                                                              score_func=lambda x: 1 / torch.norm(x, 1, 1, keepdim=True))]
    ]

    # tabulate results
    print("")
    table = tabulate(results, headers=headers, tablefmt='simple_grid')
    print(table)

    if False: # example for extracting uncertainty values
        print("")
        headers = ["", "Model", "Data", "Distr.", "filename"]
        entropy = lambda x: (2905.9861160031696 + 0.5 * torch.sum(torch.log(x))).item()
        unc_vals = [
            ["Query (Q)", *get_uncertainty_values(data, 1003, "Q", entropy)],
            ["Positive (G)", *get_uncertainty_values(data, 7146, "G", entropy)],
            ["Negative (G)", *get_uncertainty_values(data, 10000, "G", entropy)]
        ]

        print(tabulate(unc_vals, headers, tablefmt='simple_grid'))
    