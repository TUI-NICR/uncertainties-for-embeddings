import os
import json
import concurrent.futures
import random
import shutil

import torch
import torch.nn.functional as F

import numpy as np

import matplotlib.pyplot as plt

import cv2
from sklearn.neighbors import KernelDensity

from tqdm import tqdm

MEAN_VEC = 'mean_vector'
VAR_OF_MEAN = 'variance_of_mean_vector'
VAR_VEC = 'variance_vector'
VAR_OF_VAR = 'variance_of_variance_vector'

# TODO: add section headers for functions that cause console output, so it is easier to see what is what in console.
# TODO: maybe add tqdm

# NIKR:
#DATA_FILENAME = '/results_nas/ange8547/raw_model_outputs.json'
#GALLERY_PATH = '/local/meisenba/dataset_reid/Market-1501-v15.09.15/bounding_box_test'
#QUERY_PATH = '/local/meisenba/dataset_reid/Market-1501-v15.09.15/query'

# Codium:
#DATA_FILENAME = 'C:/Users/AGebh/Downloads/display_kit/uncertainty_ranking_visualization/raw_model_outputs.json'
#GALLERY_PATH = 'C:/Users/AGebh/Downloads/display_kit/uncertainty_ranking_visualization/Market-1501-v15.09.15/bounding_box_test'
#QUERY_PATH = 'C:/Users/AGebh/Downloads/display_kit/uncertainty_ranking_visualization/Market-1501-v15.09.15/query'

# HPC:
RAW_DATA_PATHS = ["/usr/scratch4/angel8547/results/UAL/" + str(i) + "/raw_model_outputs.json" for i in range(11,20)] + ["/usr/scratch4/angel8547/results/UAL/" + str(i) + "/raw_model_outputs.json" for i in range(21,30)] + ["/usr/scratch4/angel8547/results/UAL/" + str(i) + "/raw_model_outputs.json" for i in range(43,45)]
GALLERY_PATH = '/usr/scratch4/angel8547/datasets/Market1501/bounding_box_test'
QUERY_PATH = '/usr/scratch4/angel8547/datasets/Market1501/query'
DATA_FILENAME = "/usr/scratch4/angel8547/results/UAL/28/raw_model_outputs.json" # example run

assert len(RAW_DATA_PATHS) == 20

plt.rc('text', usetex=True) # latex in plots
plt.rcParams['text.latex.preamble']=r"""\usepackage{bm}
\usepackage{graphicx}
"""

bbox_dat = dict(facecolor='#fae790', edgecolor='none')
bbox_dis = dict(facecolor='#f1b3ff', edgecolor='none')
bbox_mod = dict(facecolor='#b1e6ff', edgecolor='none')
bbox_fv = dict(facecolor='#b6ffb9', edgecolor='none')
no_bbox = dict(facecolor='none', edgecolor='none')


NUM_STDS_FOR_CUTOFF = 1.5
NUM_STDS_FOR_CUTOFF_LARGE = 1.5

DATA_UNC_MEAN_NORM = 36.743523011051
DATA_UNC_NORM_STD = 0.36616902432877674
DIST_UNC_MEAN_NORM = 0.010726277209558023
DIST_UNC_NORM_STD = 0.002474199714882866
DIST_CUTOFF = DIST_UNC_MEAN_NORM + NUM_STDS_FOR_CUTOFF * DIST_UNC_NORM_STD
DATA_CUTOFF = DATA_UNC_MEAN_NORM + NUM_STDS_FOR_CUTOFF * DATA_UNC_NORM_STD
DIST_CUTOFF_LARGE = DIST_UNC_MEAN_NORM + NUM_STDS_FOR_CUTOFF_LARGE * DIST_UNC_NORM_STD
DIST_CUTOFF_STR = r"{}".format(f"{DIST_CUTOFF:.3f}")
DATA_CUTOFF_STR = r"{}".format(f"{DATA_CUTOFF:.2f}")
DIST_CUTOFF_LARGE_STR = r"{}".format(f"{DIST_CUTOFF_LARGE:.3f}")


NUM_SAMPLES_IN_SET = {
    "Q": 3368,
    "D1": 278,
    "D2": 1729,
    "D3": 567,
    "D4": 224
}


def kde2D(x, y, x_min, x_max, y_min, y_max, bandwidth=None,
          x_bins=100j, y_bins=100j): 
    """Build 2D kernel density estimate (KDE). 
    
    Return x, y, z values of probability density function (PDF). PDF value at (x,y) is z.
    """

    print("KDE2D")

    # create grid of sample locations (default: 100x100)
    xx, yy = np.mgrid[x_min:x_max:x_bins, 
                      y_min:y_max:y_bins]

    xy_sample = np.vstack([xx.ravel(), yy.ravel()])
    xy_train = np.vstack([x,y])

    if bandwidth is None:
        d = xy_train.shape[0]
        n = xy_train.shape[1]
        bandwidth = (n * (d + 2) / 4.)**(-1. / (d + 4)) # silverman
    
    kde = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
    kde.fit(xy_train.T)

    # score_samples() returns the log-likelihood of the samples
    z = np.exp(kde.score_samples(xy_sample.T))
    return xx, yy, np.reshape(z, xx.shape)


def get_vectors(data, set_id, vector_type):
    """Returns the requested raw model output vectors for the set in a list."""
    if vector_type == "fDATA": # filtered data uncertainty vectors whose images do not also have high distributional uncertainty
        return [np.array(data['data'][name][VAR_VEC])
                for name in sorted(data['sets'][set_id])
                #if np.linalg.norm(np.array(data['data'][name][VAR_OF_VAR]), 2) < DIST_CUTOFF]
                if 1024 * np.log(2 * np.pi * np.e) + 0.5 * np.sum(np.log(np.array(data['data'][name][VAR_OF_VAR]))) < -6080]
    elif vector_type == "fDIST": # filtered distributional uncertainty vectors whose images do not also have high data uncertainty
        return [np.array(data['data'][name][VAR_OF_VAR])
                for name in sorted(data['sets'][set_id])
                #if np.linalg.norm(np.array(data['data'][name][VAR_VEC]), 2) < DATA_CUTOFF]
                if 1024 * np.log(2 * np.pi * np.e) + 0.5 * np.sum(np.log(np.array(data['data'][name][VAR_VEC]))) < 2705]
    elif vector_type == "fDATA+": # filtered data uncertainty vectors whose images do not also have low data uncertainty
        return [np.array(data['data'][name][VAR_VEC])
                for name in sorted(data['sets'][set_id])
                #if np.linalg.norm(np.array(data['data'][name][VAR_OF_VAR]), 2) > DIST_CUTOFF_LARGE]
                if 1024 * np.log(2 * np.pi * np.e) + 0.5 * np.sum(np.log(np.array(data['data'][name][VAR_OF_VAR]))) > -5800]
    return [data['data'][name][vector_type]
            for name in sorted(data['sets'][set_id])]

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


def normalized(vec):
    """Return the normalized vector."""
    return vec / np.linalg.norm(vec)


def get_mean_precision_per_gallery_sample(
        dist_mat, gallery_labels, query_labels,
        query_camera_ids, gallery_camera_ids):
    """Computes mean precision for each gallery sample based on distance
    matrix. Identical calculation as for mAP, but precision for gallery
    instead of query."""
    precision_and_sorted_matches = precisions(
        dist_mat, query_labels, gallery_labels,
        query_camera_ids, gallery_camera_ids)
    
    prec = np.zeros(dist_mat.shape[1])
    cnt = np.zeros(dist_mat.shape[1])
    for precision, sorted_matches in precision_and_sorted_matches:
        prec[sorted_matches] += precision
        cnt[sorted_matches] += np.ones_like(sorted_matches)
    
    offset = np.sum(cnt == 0)
    cnt[cnt == 0] = 1
    mean_prec = prec / cnt

    non_distractor_prec = mean_prec[offset:]
    prec_sort_idx = np.argsort(non_distractor_prec) + offset
    
    return mean_prec, prec_sort_idx


def show_samples_for_id(data, id_, prec_threshold1=0.1, prec_threshold2=0.5,
                        dist_mat=None,
                        mean_prec=None, prec_sort_idx=None,
                        save_instead_of_show=True):
    """Plots images for each identity grouped by precision.

    For every identity, all the query images are plotted. The gallery images for that identity
    are split into three groups by precision: <0.3, 0.3 - 0.7, >0.7.

    filename: 'samples_{}.png'.format(id_)
    """
    FONT_SIZE = 8
    
    query_vectors = torch.Tensor(get_vectors(data, 'Q', MEAN_VEC))
    gallery_vectors = torch.Tensor(get_vectors(data, 'G', MEAN_VEC))
    if dist_mat is None:
        dist_mat = compute_cosine_distance(query_vectors, gallery_vectors)

    query_labels = get_labels(data, 'Q')
    gallery_labels = get_labels(data, 'G')

    if mean_prec is None or prec_sort_idx is None:
        mean_prec, prec_sort_idx = get_mean_precision_per_gallery_sample(
            dist_mat, gallery_labels, query_labels)
    
    gallery_imagenames = sorted(data['sets']['G'])
    query_imagenames = sorted(data['sets']['Q'])
    
    q_idxs = np.arange(dist_mat.shape[0], dtype=int)
    g_idxs = np.arange(dist_mat.shape[1], dtype=int)
    q_id_x = q_idxs[query_labels == id_]
    g_id_x = g_idxs[np.logical_and(gallery_labels == id_,
                                   mean_prec < prec_threshold1)]
    g_id_xm = g_idxs[np.logical_and(np.logical_and(
        gallery_labels == id_, mean_prec >= prec_threshold1),
        mean_prec < prec_threshold2)]
    g_id_xn = g_idxs[np.logical_and(
        gallery_labels == id_, mean_prec >= prec_threshold2)]
    
    len_q_id_x = min(len(q_id_x), 12)
    len_g_id_x = min(len(g_id_x), 12)
    len_g_id_xm = min(len(g_id_xm), 12)
    len_g_id_xn = min(len(g_id_xn), 12)
    x_axis_len = np.max([len_q_id_x, len_g_id_x, len_g_id_xm, len_g_id_xn])

    fig, axs = plt.subplots(4, x_axis_len)
    for i in range(len_q_id_x):
        q_idx = q_id_x[i]
        q_img_filename = os.path.join(QUERY_PATH,
                                      query_imagenames[q_idx])
        q_img = cv2.imread(q_img_filename)
        q_img = cv2.cvtColor(q_img, cv2.COLOR_BGR2RGB)
        axs[0, i].imshow(q_img)
        axs[0, i].tick_params(left=False, right=False, labelleft=False, 
                              labelbottom=False, bottom=False)
        if i == 0:
            axs[0, i].set_title('id={}'.format(query_labels[q_idx]),
                                fontsize=FONT_SIZE)
            axs[0, i].set_ylabel('query', fontsize=FONT_SIZE)
    for i in range(len_q_id_x, x_axis_len):
        axs[0, i].tick_params(left=False, right=False, labelleft=False, 
                           labelbottom=False, bottom=False)
        axs[0, i].axis('off')
    for i in range(len_g_id_xn):
        high_prec_idx = g_id_xn[i]
        g_img_filename = os.path.join(GALLERY_PATH,
                                      gallery_imagenames[high_prec_idx])
        g_img = cv2.imread(g_img_filename)
        g_img = cv2.cvtColor(g_img, cv2.COLOR_BGR2RGB)
        axs[1, i].imshow(g_img)
        axs[1, i].tick_params(left=False, right=False, labelleft=False, 
                              labelbottom=False, bottom=False) 
        if i == 0:
            axs[1, i].set_ylabel('gallery', fontsize=FONT_SIZE)
        axs[1, i].set_title('{}'.format(high_prec_idx), fontsize=FONT_SIZE)
        axs[1, i].set_xlabel('{:.4f}'.format(mean_prec[high_prec_idx]),
                             fontsize=FONT_SIZE)
    for i in range(len_g_id_xn, x_axis_len):
        axs[1, i].tick_params(left=False, right=False, labelleft=False, 
                           labelbottom=False, bottom=False)
        axs[1, i].axis('off')
    for i in range(len_g_id_xm):
        high_prec_idx = g_id_xm[i]
        g_img_filename = os.path.join(GALLERY_PATH,
                                      gallery_imagenames[high_prec_idx])
        g_img = cv2.imread(g_img_filename)
        g_img = cv2.cvtColor(g_img, cv2.COLOR_BGR2RGB)
        axs[2, i].imshow(g_img)
        axs[2, i].tick_params(left=False, right=False, labelleft=False, 
                           labelbottom=False, bottom=False) 
        if i == 0:
            axs[2, i].set_ylabel('< {}'.format(prec_threshold2),
                                 fontsize=FONT_SIZE)
        axs[2, i].set_title('{}'.format(high_prec_idx), fontsize=FONT_SIZE)
        axs[2, i].set_xlabel('{:.4f}'.format(mean_prec[high_prec_idx]),
                             fontsize=FONT_SIZE)
    for i in range(len_g_id_xm, x_axis_len):
        axs[2, i].tick_params(left=False, right=False, labelleft=False, 
                           labelbottom=False, bottom=False)
        axs[2, i].axis('off')
    for i in range(len_g_id_x):
        low_prec_idx = g_id_x[i]
        g_img_filename = os.path.join(GALLERY_PATH,
                                      gallery_imagenames[low_prec_idx])
        g_img = cv2.imread(g_img_filename)
        g_img = cv2.cvtColor(g_img, cv2.COLOR_BGR2RGB)
        axs[3, i].imshow(g_img)
        axs[3, i].tick_params(left=False, right=False, labelleft=False, 
                           labelbottom=False, bottom=False) 
        if i == 0:
            axs[3, i].set_ylabel('< {}'.format(prec_threshold1),
                                 fontsize=FONT_SIZE)
        axs[3, i].set_title('{}'.format(low_prec_idx), fontsize=FONT_SIZE)
        axs[3, i].set_xlabel('{:.4f}'.format(mean_prec[low_prec_idx]),
                             fontsize=FONT_SIZE)
    for i in range(len_g_id_x, x_axis_len):
        axs[3, i].tick_params(left=False, right=False, labelleft=False, 
                           labelbottom=False, bottom=False)
        axs[3, i].axis('off')
    plt.tight_layout()
    if save_instead_of_show:
        plt.savefig('samples_{}.png'.format(id_), dpi=600)
        plt.close(fig)
    else:
        plt.show()


def plot_feature_vector(data, qi=1003, gi1=7143, gi2=7146):
    """Plot feature vector for three example images, one of them a query image,
    two of them gallery images. Can be used to compare good matching with bad
    matching examples. The parameters qi, gi1, and gi2 are the indexes of the
    images in the sorted filenames of the respective sets.
    
    filename: plt.savefig('samples_Q{}_G{}_G{}.png'.format(qi, gi1, gi2), dpi=75)"""
    
    print("plot_feature_vector")
    # first, plot sample images
    gallery_imagenames = sorted(data['sets']['G'])
    query_imagenames = sorted(data['sets']['Q'])
    fig, axs = plt.subplots(1, 3)
    img_filenames = [os.path.join(QUERY_PATH, query_imagenames[qi]),
                     os.path.join(GALLERY_PATH, gallery_imagenames[gi1]),
                     os.path.join(GALLERY_PATH, gallery_imagenames[gi2])]
    colors = ['rs', 'gs', 'bs']
    for i, img_filename in enumerate(img_filenames):
        img = cv2.imread(img_filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        axs[i].imshow(img)
        axs[i].tick_params(left=False, right=False, labelleft=False, 
                              labelbottom=False, bottom=False)
        axs[i].plot([4], [4], colors[i], markersize=20)
    plt.tight_layout()
    plt.savefig('samples_Q{}_G{}_G{}.png'.format(qi, gi1, gi2), dpi=75)
    plt.close(fig)
    
    query_vectors = np.array(get_vectors(data, 'Q', MEAN_VEC))
    gallery_vectors = np.array(get_vectors(data, 'G', MEAN_VEC))
    
    # then plot feature vector with 50 dimension in each image (normalized)
    data_uncert_vecs = get_vectors(data, 'G', VAR_OF_MEAN)
    q_data_uncertainty = np.array(get_vectors(data, 'Q', VAR_OF_MEAN)[qi])
    g_data_uncertainty1 = np.array(data_uncert_vecs[gi1])
    g_data_uncertainty2 = np.array(data_uncert_vecs[gi2])
    q_vec = query_vectors[qi]
    g_vec1 = gallery_vectors[gi1]
    g_vec2 = gallery_vectors[gi2]
    g_norm1 = np.linalg.norm(g_vec1)
    g_norm2 = np.linalg.norm(g_vec2)
    g_vec_norm1 = g_vec1 / g_norm1
    g_vec_norm2 = g_vec2 / g_norm2
    q_norm = np.linalg.norm(q_vec)
    q_vec_norm = q_vec / q_norm
    g_data_uncertainty_norm1 = np.sqrt(g_data_uncertainty1) / g_norm1
    g_data_uncertainty_norm2 = np.sqrt(g_data_uncertainty2) / g_norm2
    q_data_uncertainty_norm = np.sqrt(q_data_uncertainty) / q_norm
    absmax = np.max(np.abs([q_vec_norm, g_vec_norm1, g_vec_norm2]))
    for o in range(0, len(q_vec_norm), 50):
        if o != 450:
            continue # only need an example for paper
        fig = plt.figure()
        m = min(o+50, len(q_vec_norm))
        plt.plot([o, m], [0,0], "k-", linewidth=0.5)
        plt.plot(np.arange(o, m), q_vec_norm[o:m], 'r.')
        plt.plot(np.arange(o, m), g_vec_norm1[o:m], 'g.')
        plt.plot(np.arange(o, m), g_vec_norm2[o:m], 'b.')
        for c in range(o, m):
            plt.plot([c, c],
                     [q_vec_norm[c] - q_data_uncertainty_norm[c],
                      q_vec_norm[c] + q_data_uncertainty_norm[c]], 'm-')
            plt.plot([c, c],
                     [g_vec_norm1[c] - g_data_uncertainty_norm1[c],
                      g_vec_norm1[c] + g_data_uncertainty_norm1[c]], 'c-')
            plt.plot([c, c],
                     [g_vec_norm2[c] - g_data_uncertainty_norm2[c],
                      g_vec_norm2[c] + g_data_uncertainty_norm2[c]], 'c-')
            plt.plot([c+0.5, c+0.5], [-absmax, absmax], 'k-', linewidth=0.5, alpha=0.5)
        #plt.show()
        plt.xlabel(r'Index $i$', fontsize=20)
        plt.ylabel(r'Feature Vector \raisebox{0.65ex}{\makebox[0pt][r]{\rule{0pt}{0.15em}}$\frac{\mu_i}{\left\| \mu \right\|}$}', fontsize=20, bbox=bbox_fv, labelpad=7)
        plt.tight_layout()
        plt.savefig('feat_vec_Q{}_G{}_G{}_{}.pdf'.format(qi, gi1, gi2, o),
                    dpi=600)
        plt.close(fig)
        
        
def distribution_plot(x_values, y_values, x_min=None, x_max=None, y_min=None,
                      y_max=None, title='', xlabel='', ylabel='',
                      filename=None, save_instead_of_show=True, y_bbox=no_bbox, x_bbox=no_bbox, fontsize=20):
    """Scatter plot of (x,y) values, 2D KDE PDF as contour plot, and trendline for this distribution."""
    print("distribution_plot")
    x_values = np.array(x_values)
    y_values = np.array(y_values)
    
    if x_min is None or x_max is None:
        x_min = np.min(x_values)
        x_max = np.max(x_values)
        diff_x = x_max - x_min
        x_min -= 0.03 * diff_x
        x_max += 0.03 * diff_x
    else:
        diff_x = x_max - x_min
    if y_min is None or y_max is None:
        y_min = np.min(y_values)
        y_max = np.max(y_values)
        diff_y = y_max - y_min
        y_min -= 0.03 * diff_y
        y_max += 0.03 * diff_y
    else:
        diff_y = y_max - y_min

    if filename is None:
        save_instead_of_show = False

    if save_instead_of_show:
        fig = plt.figure()
    plt.plot(x_values, y_values,
             '.', color=(0.6, 0.6, 0.6), linewidth=0.25)
    plt.title(title)
    plt.xlabel(xlabel, fontsize=fontsize, bbox=x_bbox, labelpad=7)
    plt.ylabel(ylabel, fontsize=fontsize, bbox=y_bbox, labelpad=7)
    plt.axis([x_min, x_max, y_min, y_max])

    # 2D KDE over data, plot resulting PDF as contour-plot
    y_fac = (diff_x / diff_y) * 0.75
    xx, yy, zz = kde2D(x_values, y_values*y_fac, x_min, x_max,
                       y_min*y_fac, y_max*y_fac, bandwidth=diff_x*0.065)
    yy /= y_fac
    plt.contour(xx, yy, zz, levels=[0.004, 0.01, 0.03, 0.1, 0.3, 0.7, 0.95],
                linewidths=0.5, colors='k')
    
    #print("polyfit")
    z = np.polyfit(x_values, y_values, 1)
    p = np.poly1d(z)
    x_line = np.linspace(x_min, x_max, 2)
    plt.plot(x_line, p(x_line), 'b-')
    if save_instead_of_show:
        plt.tight_layout()
        plt.savefig(filename, dpi=600)
        plt.close(fig)
    else:
        plt.show()


def distribution_1d_plot(x_values, x_min=None, x_max=None, title='',
                         xlabel='', ylabel='Probability Density',
                         filename=None, save_instead_of_show=True):
    """1D KDE to get PDF over x values. Mean is also marked."""
    print("distribution_1d_plot")
    x_mean = np.mean(x_values)
    x_std = np.std(x_values)
    if x_min is None or x_max is None:
        x_min = np.min(x_values)
        x_max = np.max(x_values)
        diff_x = x_max - x_min
    else:
        diff_x = x_max - x_min  
    
    kde = KernelDensity(bandwidth=(diff_x*0.03), kernel='gaussian')
    kde.fit(np.array(x_values)[:, np.newaxis])
    x = np.linspace(x_mean - 3.5 * x_std,
                    x_mean + 3.5 * x_std, 100)
    y = np.exp(kde.score_samples(x[:, np.newaxis]))
    
    if filename is None:
        save_instead_of_show = False

    if save_instead_of_show:
        fig = plt.figure()
    plt.plot(x, y, 'k-')
    plt.plot([x_mean, x_mean], [0, np.max(y) * 0.05], 'k-')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.axis([x_mean - 3.5 * x_std, x_mean + 3.5 * x_std, 0, np.max(y) * 1.03])
    if save_instead_of_show:
        plt.tight_layout()
        plt.savefig(filename, dpi=600)
        plt.close(fig)
    else:
        plt.show()


def plot_exemplary_correlation(data, uncertainty_type,
                               qi=1003, gi1=7143, gi2=7146,
                               save_instead_of_show=True):
    """plots correlation between absolute value of feature vector and data
    uncertainty vector (or square root of data uncertainty vector) (num_elements = num_samples).
    
    6 plots: scatter/trend/contour plot where x=abs(feature_vector) and y is 
    the given uncertainty type or its square root, for the three given images (Q,G1,G2) (2*3=6)

    filename: correlation{_sqrt/""}_{uncertainty_type_name}_Q{index}.png
    """

    q_uncertainty = np.array(get_vectors(data, 'Q', uncertainty_type)[qi])
    g_uncertainty_vecs = get_vectors(data, 'G', uncertainty_type)
    g_uncertainty1 = np.array(g_uncertainty_vecs[gi1])
    g_uncertainty2 = np.array(g_uncertainty_vecs[gi2])

    sqrt_g_uncertainty1 = np.sqrt(g_uncertainty1)
    sqrt_g_uncertainty2 = np.sqrt(g_uncertainty2)
    sqrt_q_uncertainty = np.sqrt(q_uncertainty)
    
    query_vectors = np.array(get_vectors(data, 'Q', MEAN_VEC))
    gallery_vectors = np.array(get_vectors(data, 'G', MEAN_VEC))
    q_vec = query_vectors[qi]
    g_vec1 = gallery_vectors[gi1]
    g_vec2 = gallery_vectors[gi2]
    
    vec_max = np.max([q_vec, g_vec1, g_vec2])
    vec_max += 0.03 * vec_max
    unc_max = np.max([q_uncertainty, g_uncertainty1, g_uncertainty2])
    unc_min = np.min([q_uncertainty, g_uncertainty1, g_uncertainty2])
    diff_unc = unc_max - unc_min
    unc_max += 0.03 * diff_unc
    unc_min -= 0.03 * diff_unc
    sqrt_unc_max = np.max([sqrt_q_uncertainty,
                           sqrt_g_uncertainty1,
                           sqrt_g_uncertainty2])
    sqrt_unc_min = np.min([sqrt_q_uncertainty,
                           sqrt_g_uncertainty1,
                           sqrt_g_uncertainty2])
    diff_sqrt_unc = sqrt_unc_max - sqrt_unc_min
    sqrt_unc_max += 0.03 * diff_sqrt_unc
    sqrt_unc_min -= 0.03 * diff_sqrt_unc
    
    if uncertainty_type == VAR_VEC:
        uncertainty_type_name = 'data_uncertainty'
        uncertainty_type_latex = r'\Sigma^{(D)}_i'
        uncertainty_type_latex_name = r'Data Uncertainty '
        uncertainty_type_bbox = bbox_dat
    elif uncertainty_type == VAR_OF_MEAN:
        uncertainty_type_name = 'model_uncertainty'
        uncertainty_type_latex = r'\Sigma^{(M)}_i'
        uncertainty_type_latex_name = r'Model Uncertainty '
        uncertainty_type_bbox = bbox_mod
    elif uncertainty_type == VAR_OF_VAR:
        uncertainty_type_name = 'distributional_uncertainty'
        uncertainty_type_latex_name = r'Distributional Uncertainty '
        uncertainty_type_latex = r'\Sigma^{(V)}_i'
        uncertainty_type_bbox = bbox_dis

    abs_fv_latex = r'Feature Vector $\vert \mu_i \vert$'

    # 6 plots: scatter/trend/contour plot where x=abs(feature_vector) and y is 
    # the given uncertainty type or its square root, for the three given images (Q,G1,G2)
    title = f'G{gi1}; correlation $= {np.corrcoef([np.abs(g_vec1), g_uncertainty1])[0, 1]:.4f}$'
    filename = 'correlation_{}_G{}.pdf'.format(uncertainty_type_name, gi1)
    distribution_plot(np.abs(g_vec1), g_uncertainty1,
                      x_min=0, x_max=vec_max, y_min=unc_min, y_max=unc_max,
                      title=title, xlabel=abs_fv_latex,
                      ylabel= uncertainty_type_latex_name + r'$' + uncertainty_type_latex + r'$', filename=filename,
                      save_instead_of_show=save_instead_of_show, y_bbox=uncertainty_type_bbox, x_bbox=bbox_fv)
    
    title = f'G{gi1}; correlation $= {np.corrcoef([np.abs(g_vec1), sqrt_g_uncertainty1])[0, 1]:.4f}$'
    filename = 'correlation_sqrt_{}_G{}.pdf'.format(uncertainty_type_name, gi1)
    distribution_plot(np.abs(g_vec1), sqrt_g_uncertainty1,
                      x_min=0, x_max=vec_max,
                      y_min=sqrt_unc_min, y_max=sqrt_unc_max,
                      title=title, xlabel=abs_fv_latex,
                      ylabel= uncertainty_type_latex_name + r'$\sqrt{' + uncertainty_type_latex + r'}$', filename=filename,
                      save_instead_of_show=save_instead_of_show, y_bbox=uncertainty_type_bbox, x_bbox=bbox_fv)

    title = f'G{gi2}; correlation $= {np.corrcoef([np.abs(g_vec2), g_uncertainty2])[0, 1]:.4f}$'
    filename = 'correlation_{}_G{}.pdf'.format(uncertainty_type_name, gi2)
    distribution_plot(np.abs(g_vec2), g_uncertainty2,
                      x_min=0, x_max=vec_max, y_min=unc_min, y_max=unc_max,
                      title=title, xlabel=abs_fv_latex,
                      ylabel= uncertainty_type_latex_name + r'$' + uncertainty_type_latex + r'$', filename=filename,
                      save_instead_of_show=save_instead_of_show, y_bbox=uncertainty_type_bbox, x_bbox=bbox_fv)

    title = f'G{gi2}; correlation $= {np.corrcoef([np.abs(g_vec2), sqrt_g_uncertainty2])[0, 1]:.4f}$'
    filename = 'correlation_sqrt_{}_G{}.pdf'.format(uncertainty_type_name, gi2)
    distribution_plot(np.abs(g_vec2), sqrt_g_uncertainty2,
                      x_min=0, x_max=vec_max, y_min=sqrt_unc_min,
                      y_max=sqrt_unc_max,
                      title=title, xlabel=abs_fv_latex,
                      ylabel= uncertainty_type_latex_name + r'$\sqrt{' + uncertainty_type_latex + r'}$', filename=filename,
                      save_instead_of_show=save_instead_of_show, y_bbox=uncertainty_type_bbox, x_bbox=bbox_fv)

    title = f'Q{qi}; correlation $= {np.corrcoef([np.abs(q_vec), q_uncertainty])[0, 1]:.4f}$'
    filename = 'correlation_{}_Q{}.pdf'.format(uncertainty_type_name, qi)
    distribution_plot(np.abs(q_vec), q_uncertainty,
                      x_min=0, x_max=vec_max, y_min=unc_min, y_max=unc_max,
                      title=title, xlabel=abs_fv_latex,
                      ylabel= uncertainty_type_latex_name + r'$' + uncertainty_type_latex + r'$', filename=filename,
                      save_instead_of_show=save_instead_of_show, y_bbox=uncertainty_type_bbox, x_bbox=bbox_fv)

    title = f'Q{qi}; correlation $= {np.corrcoef([np.abs(q_vec), sqrt_q_uncertainty])[0, 1]:.4f}$'
    filename = 'correlation_sqrt_{}_Q{}.pdf'.format(uncertainty_type_name, qi)
    distribution_plot(np.abs(q_vec), sqrt_q_uncertainty,
                      x_min=0, x_max=vec_max, y_min=sqrt_unc_min,
                      y_max=sqrt_unc_max,
                      title=title, xlabel=abs_fv_latex,
                      ylabel= uncertainty_type_latex_name + r'$\sqrt{' + uncertainty_type_latex + r'}$', filename=filename,
                      save_instead_of_show=save_instead_of_show, y_bbox=uncertainty_type_bbox, x_bbox=bbox_fv)


def correlation(data, uncertainty_type, subset='Q', index=0, use_sqrt=False):
    """correlation between absolute value of feature vector and uncertainty
    vector (or square root of uncertainty vector)"""

    # probably returns a float?
    
    uncertainty = np.array(get_vectors(data, subset, uncertainty_type)[index])
    feature_vector = np.array(get_vectors(data, subset, MEAN_VEC)[index])
    if use_sqrt:
        sqrt_uncertainty = np.sqrt(uncertainty)
        return np.corrcoef([np.abs(feature_vector), sqrt_uncertainty])[0, 1]
    else:
        return np.corrcoef([np.abs(feature_vector), uncertainty])[0, 1]


def correlations(data, uncertainty_type, subset, use_sqrt):
    n_samples = len(get_labels(data, subset))
    correlations = np.empty(n_samples, dtype=float)
    for i in range(n_samples):
        correlations[i] = correlation(data, uncertainty_type, subset,
                                      i, use_sqrt)
    return correlations


def compute_correlations(data):
    """Print the pearson correlation coefficient averaged over the set for all
    uncertainty types, Q+G sets, direct (var) or using sqrt(unc) (std) between
    abs(feat_vec) and the uncertainty (num_elements in vector = num_samples for PCC)."""

    rv = {}
    for uncertainty_type in [VAR_VEC, VAR_OF_MEAN, VAR_OF_VAR]:
        rv[uncertainty_type] = {}

        uncertainty_type_name = {
            VAR_VEC: "data_uncertainty",
            VAR_OF_VAR: "distributional_uncertainty",
            VAR_OF_MEAN: "model_uncertainty"
        }[uncertainty_type]

        for subset in ['Q', 'G']:
            rv[uncertainty_type][subset] = {}
            for use_sqrt in [True, False]:
                unc_type = 'std' if use_sqrt else 'var'
                corr = correlations(data, uncertainty_type, subset, use_sqrt)
                print(uncertainty_type_name, subset, unc_type,
                      np.mean(corr), np.std(corr))
                rv[uncertainty_type][subset][use_sqrt] = np.mean(corr)
    return rv


def plot_correlation_distribution(data, save_instead_of_show=True):
    """Generates 1D KDE PDF plots over correlation values between 
    abs(feat_vec) and the uncertainty (num_elements in vector = num_samples for PCC),
    for all uncertainty types, Q+G sets, direct (var) or using sqrt(unc) (std).
    
    3*2*2 = 12 plots. 
    filename: dataset_correlation_{uncertainty_type_name}_{std/var}_{Q/G}.png"""

    for uncertainty_type in [VAR_VEC, VAR_OF_MEAN, VAR_OF_VAR]:
        
        uncertainty_type_name = {
            VAR_VEC: "data_uncertainty",
            VAR_OF_VAR: "distributional_uncertainty",
            VAR_OF_MEAN: "model_uncertainty"
        }[uncertainty_type]

        for subset in ['Q', 'G']:
            for use_sqrt in [True, False]:
                unc_type = 'std' if use_sqrt else 'var'
                corr = correlations(data, uncertainty_type, subset, use_sqrt)
                
                title = '{}, {}, {}'.format(uncertainty_type_name, unc_type, subset)
                filename = 'dataset_correlation_{}_{}_{}.pdf'.format(
                    uncertainty_type_name, unc_type, subset)
                distribution_1d_plot(corr, title=title, xlabel='correlation',
                                     filename=filename,
                                     save_instead_of_show=save_instead_of_show)


def compute_maps(data):
    """Computes the mAP like normal but with various (uncertainty) vectors as feature vectors and prints it.
    
    Computes mAP based on:
    - dist_mat(feature_vectors)
    - dist_mat(model uncertainty vectors)
    - dist_mat(sqrt(model uncertainty vectors))
    - dist_mat(feature_vectors) + dist_mat(sqrt(model uncertainty vectors))
    - 1 - ((1 - dist_mat(feature_vectors)) * (1 - dist_mat(sqrt(model uncertainty vectors))))
    - dist_mat(feature_vector / sqrt(model uncertainty))
    - dist_mat(feature_vector / (sqrt(model uncertainty) + eps))
    - dist_mat(feature_vector / (sqrt(model uncertainty) + fac * norm(sqrt(distr. unc.))))
    - dist_mat(feature_vector / (sqrt(model uncertainty) + fac * norm(sqrt(data unc.))))
    """

    query_labels = get_labels(data, 'Q')
    gallery_labels = get_labels(data, 'G')
    query_camera_ids = get_camera_ids(data, 'Q')
    gallery_camera_ids = get_camera_ids(data, 'G')

    query_vectors = torch.Tensor(get_vectors(data, 'Q', MEAN_VEC))
    gallery_vectors = torch.Tensor(get_vectors(data, 'G', MEAN_VEC))
    dist_mat = compute_cosine_distance(query_vectors, gallery_vectors)
    map_ = map_from_dist_mat(dist_mat, query_labels, gallery_labels,
                             query_camera_ids, gallery_camera_ids)
    print('mAP(feature vectors) =', map_)

    query_vectors_model_unc = torch.Tensor(
        np.array(get_vectors(data, 'Q', VAR_OF_MEAN)))
    gallery_vectors_model_unc = torch.Tensor(
        np.array(get_vectors(data, 'G', VAR_OF_MEAN)))
    dist_mat_model_unc = compute_cosine_distance(
        query_vectors_model_unc, gallery_vectors_model_unc)
    map_ = map_from_dist_mat(dist_mat_model_unc, query_labels, gallery_labels,
                             query_camera_ids, gallery_camera_ids)
    print('mAP(model uncertainty as variance) =', map_)
    
    query_vectors_model_unc_SR = torch.Tensor(
        np.sqrt(get_vectors(data, 'Q', VAR_OF_MEAN)))
    gallery_vectors_model_uncSR = torch.Tensor(
        np.sqrt(get_vectors(data, 'G', VAR_OF_MEAN)))
    dist_mat_model_uncSR = compute_cosine_distance(
        query_vectors_model_unc_SR, gallery_vectors_model_uncSR)
    map_ = map_from_dist_mat(dist_mat_model_uncSR, query_labels, gallery_labels,
                             query_camera_ids, gallery_camera_ids)
    print('mAP(model uncertainty as standard deviation) =', map_)

    dist_mat_combi = dist_mat + dist_mat_model_uncSR
    map_ = map_from_dist_mat(dist_mat_combi, query_labels, gallery_labels,
                             query_camera_ids, gallery_camera_ids)
    print('mAP(feat. vec. + model unc. as std) =', map_)
    
    dist_mat_combi2 = 1 - ((1 - dist_mat) * (1 - dist_mat_model_uncSR))
    map_ = map_from_dist_mat(dist_mat_combi2, query_labels, gallery_labels,
                             query_camera_ids, gallery_camera_ids)
    print('mAP(1 - ((1 - feat. vec.) * (1 - model unc. as std))) =', map_)

    q_mean = np.array(get_vectors(data, 'Q', MEAN_VEC))
    g_mean = np.array(get_vectors(data, 'G', MEAN_VEC))
    q_model_uncSR = np.sqrt(get_vectors(data, 'Q', VAR_OF_MEAN))
    g_model_uncSR = np.sqrt(get_vectors(data, 'G', VAR_OF_MEAN))
    query_vectors_div_unc = torch.Tensor(q_mean / q_model_uncSR)
    gallery_vectors_div_unc = torch.Tensor(g_mean / g_model_uncSR)
    dist_mat_div_unc = compute_cosine_distance(query_vectors_div_unc,
                                               gallery_vectors_div_unc)
    map_ = map_from_dist_mat(dist_mat_div_unc, query_labels, gallery_labels,
                             query_camera_ids, gallery_camera_ids)
    print('mAP(feat. vec. / model unc. as std) =', map_)

    epsilon = 0.2438
    query_vectors_div_unc2 = torch.Tensor(q_mean / (q_model_uncSR + epsilon))
    gallery_vectors_div_unc2 = torch.Tensor(g_mean / (g_model_uncSR + epsilon))
    dist_mat_div_unc2 = compute_cosine_distance(query_vectors_div_unc2,
                                                gallery_vectors_div_unc2)
    map_ = map_from_dist_mat(dist_mat_div_unc2, query_labels, gallery_labels,
                             query_camera_ids, gallery_camera_ids)
    print('mAP(feat. vec. / (model unc. as std + const)) =', map_)

    q_distr_uncSR_norm = np.linalg.norm(np.sqrt(get_vectors(data, 'Q', VAR_OF_VAR)),
                           axis=1, keepdims=True)
    g_distr_uncSR_norm = np.linalg.norm(np.sqrt(get_vectors(data, 'G', VAR_OF_VAR)),
                           axis=1, keepdims=True)
    fac = 0.36
    query_vectors_div_unc3 = torch.Tensor(q_mean / (q_model_uncSR + fac * q_distr_uncSR_norm))
    gallery_vectors_div_unc3 = torch.Tensor(g_mean / (g_model_uncSR + fac * g_distr_uncSR_norm))
    dist_mat_div_unc3 = compute_cosine_distance(query_vectors_div_unc3,
                                                gallery_vectors_div_unc3)
    map_ = map_from_dist_mat(dist_mat_div_unc3, query_labels, gallery_labels,
                             query_camera_ids, gallery_camera_ids)
    print(('mAP(feat. vec. / (model unc. as std + fac'
           ' * norm(distr. unc. as std))) ='), map_)

    q_data_uncSR_norm = np.linalg.norm(np.sqrt(get_vectors(data, 'Q', VAR_VEC)),
                          axis=1, keepdims=True)
    g_data_uncSR_norm = np.linalg.norm(np.sqrt(get_vectors(data, 'G', VAR_VEC)),
                          axis=1, keepdims=True)
    fac = 0.006
    query_vectors_div_unc4 = torch.Tensor(q_mean / (q_model_uncSR + fac * q_data_uncSR_norm))
    gallery_vectors_div_unc4 = torch.Tensor(g_mean / (g_model_uncSR + fac * g_data_uncSR_norm))
    dist_mat_div_unc4 = compute_cosine_distance(query_vectors_div_unc4,
                                                gallery_vectors_div_unc4)
    map_ = map_from_dist_mat(dist_mat_div_unc4, query_labels, gallery_labels,
                             query_camera_ids, gallery_camera_ids)
    print(('mAP(feat. vec. / (model unc. as std + fac'
           ' * norm(data unc. as std))) ='), map_)


def correlation_with_differences(data, uncertainty_type, use_sqrt=False):
    """Calculates correlation between absolute matching error and uncertainty.
    
    Returns:
        vector_correlations: how correlated is the difference in a given component of the feature vector with 
            the summed uncertainty in that position? List over all relevant Q/G pairs.
        cos_distances: list of cos distances for all relevant Q/G pairs
        unc_norms: list of sum of norm of uncertainty vectors for all relevant Q/G pairs
        norm_correlation: how correlated is the distance between the feature vectors with 
            the sum of the norm of their corresponding uncertainty vectors?
    """

    query_vectors = np.array(get_vectors(data, 'Q', MEAN_VEC))
    gallery_vectors = np.array(get_vectors(data, 'G', MEAN_VEC))

    query_uncert_vecs = np.array(get_vectors(data, 'Q', uncertainty_type))
    gallery_uncert_vecs = np.array(get_vectors(data, 'G', uncertainty_type))
    if use_sqrt:
        query_uncert_vecs = np.sqrt(query_uncert_vecs)
        gallery_uncert_vecs = np.sqrt(gallery_uncert_vecs)

    query_labels = get_labels(data, 'Q')
    gallery_labels = get_labels(data, 'G')
    
    query_camera_ids = get_camera_ids(data, 'Q')
    gallery_camera_ids = get_camera_ids(data, 'G')
    
    vector_correlations = []
    cos_distances = []
    unc_norms = []
    
    idxs = np.arange(len(gallery_labels), dtype=int)
    # iterate over query set with index, label, cid
    for qi, (q_label, cid) in enumerate(tqdm(zip(query_labels, query_camera_ids), total=len(query_labels))):
        # relevant matches as in mAP calculation
        relevant_matches = ((gallery_labels == q_label) &
                            (gallery_camera_ids != cid))
        relevant_g_indexes = idxs[relevant_matches]

        q_vec_norm = normalized(query_vectors[qi]) # normalized feature vector of current query image
        q_unc = query_uncert_vecs[qi] # uncertainty vector for current query image and specified uncertainty type

        # iterate over relevant gallery matches for the current query with index
        for gi in relevant_g_indexes:

            g_vec_norm = normalized(gallery_vectors[gi]) # normalized feature vector of current gallery image
            g_unc = gallery_uncert_vecs[gi] # uncertainty vector for current gallery image and specified uncertainty type
            diff = np.abs(q_vec_norm - g_vec_norm)
            unc = q_unc + g_unc

            # how correlated is the difference in a given component of the feature vector with the summed uncertainty in that position?
            # (for a relevant query gallery pair)
            cor_coef = np.corrcoef([diff, unc])[0, 1] 
            vector_correlations.append(cor_coef)
            
            cos_dist = 1 - q_vec_norm.dot(g_vec_norm)
            cos_distances.append(cos_dist)
            
            unc_norm = np.linalg.norm(q_unc) + np.linalg.norm(g_unc)
            unc_norms.append(unc_norm)

    # how correlated is the distance between the feature vectors with the sum of the norm of their corresponding uncertainty vectors?
    # (over all relevant query gallery pairs)
    norm_correlation = np.corrcoef([cos_distances, unc_norms])[0, 1]

    return vector_correlations, cos_distances, unc_norms, norm_correlation


def plot_uncertainty_vs_distance(data, save_instead_of_show=True):
    """Generates a bunch of plots comparing uncertainty with distance between feature vectors.
    
    - Probability Density Function (PDF) over Pearson Corellation Coefficients (PCCs) between embedding distance and data uncertainty for relevant Q/G pairs.
    - Scatter plot: cos distance vs. sum of norm of q&g data uncertainty
    - previous two again but with sqrt(uncertainty) aka std
    - same for model and distributional uncertainty
    - scatter plot of uncertainties against each other: sum of norm of q&g X uncertainty std vectors vs. sum of norm of q&g Y uncertainty std vectors (data vs model, data vs dist, model vs dist)
    - scatter plot: correlation between cos distance and corellation between sqrt(model uncertainty) and embedding distance
    """
    
    do_other_plots = False # False means: skip most plots, only do uncertainty types against each other

    # data uncertainty vs distance
    print("data uncertainty vs distance")
    if do_other_plots:
        (data_var_vector_correlations,
        cos_distances,
        data_var_unc_norms,
        data_var_norm_correlation) = correlation_with_differences(
            data, uncertainty_type=VAR_VEC, use_sqrt=False)
        distribution_1d_plot(
            data_var_vector_correlations,
            #title='data var uncertainty',
            xlabel=r'$\mathrm{corr}\left( \left\vert \frac{\mu_{\mathrm{Q}}}{\left\| \mu_{\mathrm{Q}} \right\|} - \frac{\mu_{\mathrm{G}}}{\left\| \mu_{\mathrm{G}} \right\|} \right\vert, {\Sigma^{(D)}_{\mathrm{Q}} } + {\Sigma^{(D)}_{\mathrm{G}} } \right)$',
            ylabel='Probability Density',
            filename='unc_vs_dist__data_var_vector_corr.pdf',
            save_instead_of_show=save_instead_of_show)
        distribution_plot(
            cos_distances, data_var_unc_norms,
            xlabel=r'$d_{\mathrm{cos}}(\mu_{\mathrm{Query}}, \mu_{\mathrm{Gallery}})$',
            ylabel=r'$\left\| \Sigma^{(D)}_{\mathrm{Query}} \right\| + \left\| \Sigma^{(D)}_{\mathrm{Gallery}} \right\|$',
            title=f'correlation $={data_var_norm_correlation:.4f}$',
            filename='unc_vs_cos_dist__data_var_corr.pdf',
            save_instead_of_show=save_instead_of_show, y_bbox=bbox_dat)

    (data_std_vector_correlations,
    cos_distances,
    data_std_unc_norms,
    data_std_norm_correlation) = correlation_with_differences(
        data, uncertainty_type=VAR_VEC, use_sqrt=True)
    if do_other_plots:
        distribution_1d_plot(
            data_std_vector_correlations,
            #title='data std uncertainty',
            xlabel=r'$\mathrm{corr}\left( \left\vert \frac{\mu_{\mathrm{Q}}}{\left\| \mu_{\mathrm{Q}} \right\|} - \frac{\mu_{\mathrm{G}}}{\left\| \mu_{\mathrm{G}} \right\|} \right\vert, \sqrt{\Sigma^{(D)}_{\mathrm{Q}} } + \sqrt{\Sigma^{(D)}_{\mathrm{G}} } \right)$',
            ylabel='Probability Density',
            filename='unc_vs_dist__data_std_vector_corr.pdf',
            save_instead_of_show=save_instead_of_show)
        distribution_plot(
            cos_distances, data_std_unc_norms,
            xlabel=r'$d_{\mathrm{cos}}(\mu_{\mathrm{Query}}, \mu_{\mathrm{Gallery}})$',
            ylabel=r'$\left\| \sqrt{\Sigma^{(D)}_{\mathrm{Query}}} \right\| + \left\| \sqrt{\Sigma^{(D)}_{\mathrm{Gallery}}} \right\|$',
            title=f'correlation $={data_std_norm_correlation:.4f}$',
            filename='unc_vs_cos_dist__data_std_corr.pdf',
            save_instead_of_show=save_instead_of_show, y_bbox=bbox_dat)
        
    # model uncertainty vs distance
    print("model uncertainty vs distance")
    if do_other_plots:
        (model_var_vector_correlations,
        cos_distances,
        model_var_unc_norms,
        model_var_norm_correlation) = correlation_with_differences(
            data, uncertainty_type=VAR_OF_MEAN, use_sqrt=False)
        distribution_1d_plot(
            model_var_vector_correlations,
            #title='model var uncertainty',
            xlabel=r'$\mathrm{corr}\left( \left\vert \frac{\mu_{\mathrm{Q}}}{\left\| \mu_{\mathrm{Q}} \right\|} - \frac{\mu_{\mathrm{G}}}{\left\| \mu_{\mathrm{G}} \right\|} \right\vert, {\Sigma^{(M)}_{\mathrm{Q}} } + {\Sigma^{(M)}_{\mathrm{G}} } \right)$',
            ylabel='Probability Density',
            filename='unc_vs_dist__model_var_vector_corr.pdf',
            save_instead_of_show=save_instead_of_show)
        distribution_plot(
            cos_distances, model_var_unc_norms,
            xlabel=r'$d_{\mathrm{cos}}(\mu_{\mathrm{Query}}, \mu_{\mathrm{Gallery}})$',
            ylabel=r'$\left\| \Sigma^{(M)}_{\mathrm{Query}} \right\| + \left\| \Sigma^{(M)}_{\mathrm{Gallery}} \right\|$',
            title=f'correlation $={model_var_norm_correlation:.4f}$',
            filename='unc_vs_cos_dist__model_var_corr.pdf',
            save_instead_of_show=save_instead_of_show, y_bbox=bbox_mod)

    (model_std_vector_correlations,
    cos_distances,
    model_std_unc_norms,
    model_std_norm_correlation) = correlation_with_differences(
        data, uncertainty_type=VAR_OF_MEAN, use_sqrt=True)
    if do_other_plots:
        distribution_1d_plot(
            model_std_vector_correlations,
            #title='model std uncertainty',
            xlabel=r'$\mathrm{corr}\left( \left\vert \frac{\mu_{\mathrm{Q}}}{\left\| \mu_{\mathrm{Q}} \right\|} - \frac{\mu_{\mathrm{G}}}{\left\| \mu_{\mathrm{G}} \right\|} \right\vert, \sqrt{\Sigma^{(M)}_{\mathrm{Q}} } + \sqrt{\Sigma^{(M)}_{\mathrm{G}} } \right)$',
            ylabel='Probability Density',
            filename='unc_vs_dist__model_std_vector_corr.pdf',
            save_instead_of_show=save_instead_of_show)
        distribution_plot(
            cos_distances, model_std_unc_norms,
            xlabel=r'$d_{\mathrm{cos}}(\mu_{\mathrm{Query}}, \mu_{\mathrm{Gallery}})$',
            ylabel=r'$\left\| \sqrt{\Sigma^{(M)}_{\mathrm{Query}}} \right\| + \left\| \sqrt{\Sigma^{(M)}_{\mathrm{Gallery}}} \right\|$',
            title=f'correlation $={model_std_norm_correlation:.4f}$',
            filename='unc_vs_cos_dist__model_std_corr.pdf',
            save_instead_of_show=save_instead_of_show, y_bbox=bbox_mod)
        
    # distributional uncertainty vs distance
    print("distributional uncertainty vs distance")
    if do_other_plots:
        (dis_var_vector_correlations,
        cos_distances,
        dis_var_unc_norms,
        dis_var_norm_correlation) = correlation_with_differences(
            data, uncertainty_type=VAR_OF_VAR, use_sqrt=False)
        distribution_1d_plot(
            dis_var_vector_correlations,
            #title='distributional var uncertainty',
            xlabel=r'$\mathrm{corr}\left( \left\vert \frac{\mu_{\mathrm{Q}}}{\left\| \mu_{\mathrm{Q}} \right\|} - \frac{\mu_{\mathrm{G}}}{\left\| \mu_{\mathrm{G}} \right\|} \right\vert, {\Sigma^{(V)}_{\mathrm{Q}} } + {\Sigma^{(V)}_{\mathrm{G}} } \right)$',
            ylabel='Probability Density',
            filename='unc_vs_dist__distr_var_vector_corr.pdf',
            save_instead_of_show=save_instead_of_show)
        distribution_plot(
            cos_distances, dis_var_unc_norms,
            xlabel=r'$d_{\mathrm{cos}}(\mu_{\mathrm{Query}}, \mu_{\mathrm{Gallery}})$',
            ylabel=r'$\left\| \Sigma^{(V)}_{\mathrm{Query}} \right\| + \left\| \Sigma^{(V)}_{\mathrm{Gallery}} \right\|$',
            title=f'correlation $={dis_var_norm_correlation:.4f}$',
            filename='unc_vs_cos_dist__distr_var_corr.pdf',
            save_instead_of_show=save_instead_of_show, y_bbox=bbox_dis)

    (dis_std_vector_correlations,
    cos_distances,
    dis_std_unc_norms,
    dis_std_norm_correlation) = correlation_with_differences(
        data, uncertainty_type=VAR_OF_VAR, use_sqrt=True)
    
    if do_other_plots:
        distribution_1d_plot(
            dis_std_vector_correlations,
            #title='distributional std uncertainty',
            xlabel=r'$\mathrm{corr}\left( \left\vert \frac{\mu_{\mathrm{Q}}}{\left\| \mu_{\mathrm{Q}} \right\|} - \frac{\mu_{\mathrm{G}}}{\left\| \mu_{\mathrm{G}} \right\|} \right\vert, \sqrt{\Sigma^{(V)}_{\mathrm{Q}} } + \sqrt{\Sigma^{(V)}_{\mathrm{G}} } \right)$',
            ylabel='Probability Density',
            filename='unc_vs_dist__distr_std_vector_corr.pdf',
            save_instead_of_show=save_instead_of_show)
        distribution_plot(
            cos_distances, dis_std_unc_norms,
            xlabel=r'$d_{\mathrm{cos}}(\mu_{\mathrm{Query}}, \mu_{\mathrm{Gallery}})$',
            ylabel=r'$\left\| \sqrt{\Sigma^{(V)}_{\mathrm{Query}}} \right\| + \left\| \sqrt{\Sigma^{(V)}_{\mathrm{Gallery}}} \right\|$',
            title=f'correlation $={dis_std_norm_correlation:.4f}$',
            filename='unc_vs_cos_dist__distr_std_corr.pdf',
            save_instead_of_show=save_instead_of_show, y_bbox=bbox_dis)

    # correlation between uncertainty types
    print("correlation between uncertainty types")
    cor_coef = np.corrcoef([data_std_unc_norms, dis_std_unc_norms])[0, 1]
    distribution_plot(
        data_std_unc_norms, dis_std_unc_norms,
        xlabel=r'\raisebox{-0.5em}{\shortstack{Data\\Uncertainty}} $\left\| \sqrt{\Sigma^{(D)}_{\mathrm{Query}}} \right\| + \left\| \sqrt{\Sigma^{(D)}_{\mathrm{Gallery}}} \right\|$',
        ylabel=r'\raisebox{-0.5em}{\shortstack{Distributional\\Uncertainty}} $\left\| \sqrt{\Sigma^{(V)}_{\mathrm{Query}}} \right\| + \left\| \sqrt{\Sigma^{(V)}_{\mathrm{Gallery}}} \right\|$',
        title=f'correlation $={cor_coef:.4f}$',
        filename='unc_data_norm_vs_distr_norm_corr.pdf',
        save_instead_of_show=save_instead_of_show, y_bbox=bbox_dis, x_bbox=bbox_dat, fontsize=15)

    cor_coef = np.corrcoef([data_std_unc_norms, model_std_unc_norms])[0, 1]
    distribution_plot(
        data_std_unc_norms, model_std_unc_norms,
        xlabel=r'\raisebox{-0.5em}{\shortstack{Data\\Uncertainty}} $\left\| \sqrt{\Sigma^{(D)}_{\mathrm{Query}}} \right\| + \left\| \sqrt{\Sigma^{(D)}_{\mathrm{Gallery}}} \right\|$',
        ylabel=r'\raisebox{-0.5em}{\shortstack{Model\\Uncertainty}} $\left\| \sqrt{\Sigma^{(M)}_{\mathrm{Query}}} \right\| + \left\| \sqrt{\Sigma^{(M)}_{\mathrm{Gallery}}} \right\|$',
        title=f'correlation $={cor_coef:.4f}$',
        filename='unc_data_norm_vs_model_norm_corr.pdf',
        save_instead_of_show=save_instead_of_show, y_bbox=bbox_mod, x_bbox=bbox_dat, fontsize=15)

    cor_coef = np.corrcoef([model_std_unc_norms, dis_std_unc_norms])[0, 1]
    distribution_plot(
        model_std_unc_norms, dis_std_unc_norms,
        xlabel=r'\raisebox{-0.5em}{\shortstack{Model\\Uncertainty}} $\left\| \sqrt{\Sigma^{(M)}_{\mathrm{Query}}} \right\| + \left\| \sqrt{\Sigma^{(M)}_{\mathrm{Gallery}}} \right\|$',
        ylabel=r'\raisebox{-0.5em}{\shortstack{Distributional\\Uncertainty}} $\left\| \sqrt{\Sigma^{(V)}_{\mathrm{Query}}} \right\| + \left\| \sqrt{\Sigma^{(V)}_{\mathrm{Gallery}}} \right\|$',
        title=f'correlation $={cor_coef:.4f}$',
        filename='unc_model_norm_vs_distr_norm_corr.pdf',
        save_instead_of_show=save_instead_of_show, y_bbox=bbox_dis, x_bbox=bbox_mod, fontsize=15)

    # duplicate
    """cor_coef = np.corrcoef([data_std_unc_norms, dis_std_unc_norms])[0, 1]
    distribution_plot(
        data_std_unc_norms, dis_std_unc_norms,
        xlabel='sum of norm of q&g data uncertainty std vectors',
        ylabel='sum of norm of q&g distr. uncertainty std vectors',
        title=f'correlation $={cor_coef:.4f}$',
        filename='unc_data_norm_vs_distr_norm_corr.pdf',
        save_instead_of_show=save_instead_of_show)"""
    
    # correlation between cos distance and corellation between sqrt(model uncertainty) and embedding distance
    print("correlation between cos distance and corellation between sqrt(model uncertainty) and embedding distance")
    cor_coef = np.corrcoef([cos_distances, model_std_vector_correlations])[0, 1]
    distribution_plot(
        cos_distances, model_std_vector_correlations,
        xlabel=r'$d_{\mathrm{cos}}(\mu_{\mathrm{Query}}, \mu_{\mathrm{Gallery}})$',
        ylabel=r'$\mathrm{corr}\left(\left\vert \frac{\mu_{\mathrm{Q}}}{\left\| \mu_{\mathrm{Q}} \right\|} - \frac{\mu_{\mathrm{G}}}{\left\| \mu_{\mathrm{G}} \right\|} \right\vert, \sqrt{\Sigma^{(M)}_{\mathrm{Q}} } + \sqrt{\Sigma^{(M)}_{\mathrm{G}} }\right)$',
        title=f'correlation $={cor_coef:.4f}$',
        filename='unc_vs_dist__model_vector_correlation_vs_cos_dist.pdf',
        save_instead_of_show=save_instead_of_show)


def uncertainty_vs_precision(data, uncertainty_type, use_sqrt=False,
                             save_instead_of_show=True):
    """Generates scatter plot of precision vs. norm(uncertainty_vector) for gallery samples."""

    uncertainty_type_name = {
            VAR_VEC: "data_uncertainty",
            VAR_OF_VAR: "distributional_uncertainty",
            VAR_OF_MEAN: "model_uncertainty"
        }[uncertainty_type]

    query_vectors = torch.Tensor(get_vectors(data, 'Q', MEAN_VEC))
    gallery_vectors = torch.Tensor(get_vectors(data, 'G', MEAN_VEC))
    dist_mat = compute_cosine_distance(query_vectors, gallery_vectors)
    
    query_labels = get_labels(data, 'Q')
    gallery_labels = get_labels(data, 'G')
    query_camera_ids = get_camera_ids(data, 'Q')
    gallery_camera_ids = get_camera_ids(data, 'G')
    
    mean_prec, prec_sort_idx = get_mean_precision_per_gallery_sample(
        dist_mat, gallery_labels, query_labels,
        query_camera_ids, gallery_camera_ids)

    uncert_vecs = get_vectors(data, 'G', uncertainty_type)
    if use_sqrt:
        uncert_vecs = np.sqrt(uncert_vecs)
        uncertainty_type_name = 'sqrt({})'.format(uncertainty_type_name)
        var = 'std'
    else:
        var = 'var'

    uncerts = np.array([np.linalg.norm(v) for v in uncert_vecs])
    
    title = 'correlation = {}'.format(
        np.corrcoef([mean_prec[prec_sort_idx], uncerts[prec_sort_idx]])[0, 1])
    distribution_plot(mean_prec[prec_sort_idx], uncerts[prec_sort_idx],
                      title=title, xlabel='mean precision',
                      ylabel='norm {} vec'.format(uncertainty_type_name),
                      filename='prec_vs_unc_{}_{}.pdf'.format(uncertainty_type_name, var),
                      save_instead_of_show=save_instead_of_show)
    return np.corrcoef([mean_prec[prec_sort_idx], uncerts[prec_sort_idx]])[0, 1]


def hyperparameter_search(data, hint_value=0.2438, delta=0.0001, save_instead_of_show=True, filename="hyperparameter_search.png"):
    """ Hyperparameter search with logarithmic search scale around the hint
    value and the minimum between two hyperparameter values defiend by delta.
    Use case here is the division of the feature vector by the uncertainty
    std vector with an added constant epsilon, which is the hyperparameter
    that needs to be explored.
    """

    query_labels = get_labels(data, 'Q')
    gallery_labels = get_labels(data, 'G')
    query_camera_ids = get_camera_ids(data, 'Q')
    gallery_camera_ids = get_camera_ids(data, 'G')

    q_mean = np.array(get_vectors(data, 'Q', MEAN_VEC))
    g_mean = np.array(get_vectors(data, 'G', MEAN_VEC))
    q_std = np.sqrt(get_vectors(data, 'Q', VAR_OF_MEAN))
    g_std = np.sqrt(get_vectors(data, 'G', VAR_OF_MEAN))

    factor = 2 ** (1/4)    # 4th root of 2
    iterations = int(np.log(hint_value/delta)/np.log(factor))
    sub = factor ** 10 - 1
    diff = delta * (factor ** np.arange(10, iterations+1) - sub)
    epsilon_values = sorted([0, hint_value] + list(hint_value - diff) +
                            list(hint_value + diff))
    maps = []
    for epsilon in epsilon_values:
        query_vectors_div_unc2 = torch.Tensor(q_mean / (q_std + epsilon))
        gallery_vectors_div_unc2 = torch.Tensor(g_mean / (g_std + epsilon))
        dist_mat_div_unc2 = compute_cosine_distance(query_vectors_div_unc2,
                                                    gallery_vectors_div_unc2)
        map_ = map_from_dist_mat(dist_mat_div_unc2,
                                 query_labels, gallery_labels,
                                 query_camera_ids, gallery_camera_ids)
        print('mAP(epsilon={}) = {}'.format(epsilon, map_))
        
        maps.append(map_)

    """if save_instead_of_show:
        fig = plt.figure()
    
    plt.plot(epsilon_values, maps, 'k-')
    plt.axis([0, np.max(epsilon_values), np.min(maps), np.max(maps)])
    plt.xlabel('epsilon')
    plt.ylabel('mAP')

    if save_instead_of_show:
        plt.tight_layout()
        plt.savefig(filename, dpi=600)
        plt.close(fig)
    else:
        plt.show()"""
    
    
    best_idx = np.argmax(maps)
    best_epsilon = epsilon_values[best_idx]
    best_map = maps[best_idx]
    print('best epsilon =', best_epsilon, '; mAP =', best_map)

    return best_epsilon, best_map


def hyperparameter_search2(data, uncertainty_type, hint_value=0.2438, delta=0.0001, save_instead_of_show=True, 
                           filename="hyperparameter_search.png", use_norm=True):
    """ Hyperparameter search with logarithmic search scale around the hint
    value and the minimum between two hyperparameter values defiend by delta.
    Use case here is the division of the feature vector by the uncertainty
    std vector with an added constant epsilon, which is determined through the 
    uncertainty vector, weighed by lambda which is the hyperparameter
    that needs to be explored. 
    """

    query_labels = get_labels(data, 'Q')
    gallery_labels = get_labels(data, 'G')
    query_camera_ids = get_camera_ids(data, 'Q')
    gallery_camera_ids = get_camera_ids(data, 'G')

    q_mean = np.array(get_vectors(data, 'Q', MEAN_VEC))
    g_mean = np.array(get_vectors(data, 'G', MEAN_VEC))
    q_std = np.sqrt(get_vectors(data, 'Q', VAR_OF_MEAN))
    g_std = np.sqrt(get_vectors(data, 'G', VAR_OF_MEAN))

    q_attenuation_values = np.array(get_vectors(data, 'Q', uncertainty_type))
    g_attenuation_values = np.array(get_vectors(data, 'G', uncertainty_type))

    if use_norm:
        q_attenuation_values = np.linalg.norm(q_attenuation_values, axis=1, keepdims=True)
        g_attenuation_values = np.linalg.norm(g_attenuation_values, axis=1, keepdims=True)


    factor = 2 ** (1/4)    # 4th root of 2
    iterations = int(np.log(hint_value/delta)/np.log(factor))
    sub = factor ** 10 - 1
    diff = delta * (factor ** np.arange(10, iterations+1) - sub)
    lambda_values = sorted([0, hint_value] + list(hint_value - diff) +
                            list(hint_value + diff))
    maps = []
    for lambda_ in lambda_values:
        query_vectors_div_unc2 = torch.Tensor(q_mean / (q_std + lambda_ * q_attenuation_values))
        gallery_vectors_div_unc2 = torch.Tensor(g_mean / (g_std + lambda_ * g_attenuation_values))
        dist_mat_div_unc2 = compute_cosine_distance(query_vectors_div_unc2,
                                                    gallery_vectors_div_unc2)
        map_ = map_from_dist_mat(dist_mat_div_unc2,
                                 query_labels, gallery_labels,
                                 query_camera_ids, gallery_camera_ids)
        print('mAP(lambda={}) = {}'.format(lambda_, map_))
        
        maps.append(map_)

    if save_instead_of_show:
        fig = plt.figure()
    
    plt.plot(lambda_values, maps, 'k-')
    plt.axis([0, np.max(lambda_values), np.min(maps), np.max(maps)])
    plt.xlabel('lambda')
    plt.ylabel('mAP')

    if save_instead_of_show:
        plt.tight_layout()
        plt.savefig(filename, dpi=600)
        plt.close(fig)
    else:
        plt.show()
    
    
    best_idx = np.argmax(maps)
    best_lambda = lambda_values[best_idx]
    best_map = maps[best_idx]
    print(f'using {uncertainty_type}{", no norm"* (not use_norm)}: best lambda =', best_lambda, '; mAP =', best_map)

    return best_lambda, best_map


def load_data(path="/data2/hpcuser/angel8547/results_backup_2024_03_13/UAL/28/raw_model_outputs.json"):#DATA_FILENAME):
    print(f"loading {path}...")
    with open(path, 'r') as data_file:
        data = json.load(data_file)
    print("DONE!")

    if False:
        print("Checking for non-finite values...")
        found_nf = False
        for vecs in data["data"].values():
            for k, vec in vecs.items():
                vec = np.array(vec)
                all_finite = np.isfinite(vec).all()
                if not all_finite:
                    found_nf = True
                    vec[~np.isfinite(vec)] = 0.0
                    vecs[k] = vec
        print("Done.")
        if found_nf:
            print("Found non-finite values. They have been set to 0.0.")
        else:
            print("All values are finite!")

    return data


def markus():

    # read data from file
    data = load_data()


    print("getting stuff...")
    # feature vectors and distance matrix between query and gallery features
    query_vectors = torch.Tensor(get_vectors(data, 'Q', MEAN_VEC))
    gallery_vectors = torch.Tensor(get_vectors(data, 'G', MEAN_VEC))
    dist_mat = compute_cosine_distance(query_vectors, gallery_vectors)

    # labels and camera IDs for matching
    query_labels = get_labels(data, 'Q')
    gallery_labels = get_labels(data, 'G')
    query_camera_ids = get_camera_ids(data, 'Q')
    gallery_camera_ids = get_camera_ids(data, 'G')

    # mAP = 0.8703402955540096
    """mean_average_precision = map_from_dist_mat(dist_mat,
                                               query_labels,
                                               gallery_labels,
                                               query_camera_ids,
                                               gallery_camera_ids)
    print('mean_average_precision =', mean_average_precision)"""

    # feature vector with uncertainties
    plot_feature_vector(data, qi=1003, gi1=7143, gi2=7146)

    # correlation
    plot_exemplary_correlation(data, VAR_VEC, qi=1003, gi1=7143, gi2=7146)
    plot_exemplary_correlation(data, VAR_OF_MEAN, qi=1003, gi1=7143, gi2=7146)
    plot_exemplary_correlation(data, VAR_OF_VAR, qi=1003, gi1=7143, gi2=7146)
    #compute_correlations(data)
    #plot_correlation_distribution(data)

    # uncertainty vs distance
    plot_uncertainty_vs_distance(data)
    
    # uncertainty vs precision
    for unc in [VAR_VEC, VAR_OF_MEAN, VAR_OF_VAR]:
        for use_sqrt in [False, True]:
            uncertainty_vs_precision(data, unc, use_sqrt)

    # mean average precion with different possibilities of using uncertainty
    compute_maps(data)

    # calculate precision per gallery sample --> overall: 0.8738534724399559
    mean_prec, prec_sort_idx = get_mean_precision_per_gallery_sample(
        dist_mat, gallery_labels, query_labels,
        query_camera_ids, gallery_camera_ids)
    overall_mean_precision = np.mean(mean_prec[prec_sort_idx])
    print('overall mean precision over all gallery samples = ',
          overall_mean_precision)

    # samples for each id in gallery labels sort by precision
    """for id_ in np.unique(gallery_labels):
        show_samples_for_id(data, id_, prec_threshold1=0.3,
                            prec_threshold2=0.7, dist_mat=dist_mat,
                            mean_prec=mean_prec, prec_sort_idx=prec_sort_idx)"""

#markus()

def hyperparameter_search_many_runs():

    best_epsilons = []
    best_mAPs = []
    for path in RAW_DATA_PATHS:
        print("-" * 20)
        print(f"Hyperparameter search on {path}")
        print("-" * 20)
        data = load_data(path)
        epsilon, mAP = hyperparameter_search(data, filename=f"hyper_{path.split('/')[-1]}.png")
        best_epsilons.append(epsilon)
        best_mAPs.append(mAP)
    print("")

    print(f"Best epsilons: {best_epsilons}")
    print(f"Best mAPs: {best_mAPs}")
    print(f"Mean of best epsilons: {np.mean(best_epsilons)} ±{np.std(best_epsilons)}")

    distribution_1d_plot(best_epsilons, title="Distribution of epsilon", xlabel='epsilon',
                                     filename="eps_hyper_search_distribution.png",
                                     save_instead_of_show=True)

    """fig = plt.figure()
    plt.plot(np.array(best_mAPs), np.array(best_epsilons), "x")
    plt.xlabel("mAP")
    plt.ylabel("epsilon")
    plt.title("Hyperparameter search results for different UAL runs")
    #plt.plot([x_mean, x_mean], [0, np.max(y) * 0.05], 'k-')
    plt.savefig("mAP_vs_eps.png")

    plt.close(fig)"""

#hyperparameter_search_many_runs()

# TODO: the name of the hyperparameter search plot (per search) seems broken
def main():
    for use_norm in [True, False]:
        print("-" * 20)
        print(f"Using Norm {use_norm}")
        print("-" * 20)
        print("")
        for uncertainty_type in [VAR_VEC, VAR_OF_VAR]:
            print("-" * 20)
            print(f"Uncertainty Type {uncertainty_type}")
            print("-" * 20)
            print("")

            hint_value = {
                VAR_VEC: 0.006,
                VAR_OF_VAR: 0.36
            }[uncertainty_type]

            best_lambdas = []
            best_mAPs = []
            for path in RAW_DATA_PATHS:
                print("-" * 20)
                print(f"Hyperparameter search on {path}")
                print("-" * 20)
                data = load_data(path)
                lambda_, mAP = hyperparameter_search2(data, uncertainty_type, filename=f"hyper_{uncertainty_type}_{path.split('/')[-1]}_norm_{use_norm}.png",
                                                    hint_value=hint_value, use_norm=use_norm)
                best_lambdas.append(lambda_)
                best_mAPs.append(mAP)
            print("")

            print(f"Best lambdas: {best_lambdas}")
            print(f"Mean of best lambdas: {np.mean(best_lambdas)} ±{np.std(best_lambdas)}")

            distribution_1d_plot(best_lambdas, title=f"Distribution of lambda, {uncertainty_type}{', no norm'* (not use_norm)}", xlabel='lambda',
                                            filename=f"lambda_hyper_search_distribution__{uncertainty_type}_norm_{use_norm}.png",
                                            save_instead_of_show=True)

            fig = plt.figure()
            plt.plot(np.array(best_mAPs), np.array(best_lambdas), "x")
            plt.xlabel("mAP")
            plt.ylabel("lambda")
            plt.title(f"lambda Hyperparameter search results for different UAL runs ({uncertainty_type}{', no norm'* (not use_norm)})")
            #plt.plot([x_mean, x_mean], [0, np.max(y) * 0.05], 'k-')
            plt.savefig(f"mAP_vs_lambda__{uncertainty_type}_norm_{use_norm}.png")

            plt.close(fig)


def verify_markus():

    def process_data(data):
        result = {}
        print(1)
        fv_vs_unc_correlations = compute_correlations(data) # 1
        result["fv_vs_unc_correlations"] = fv_vs_unc_correlations

        print("2/3")
        diff_vs_unc_correlations = {} # 2
        dist_vs_unc_norm_correlations = {} # 3
        for uncertainty_type in [VAR_VEC, VAR_OF_MEAN, VAR_OF_VAR]:
            diff_vs_unc_correlations[uncertainty_type] = {}
            dist_vs_unc_norm_correlations[uncertainty_type] = {}
            for use_sqrt in [False, True]:
                (vector_correlations,
                cos_distances, 
                unc_norms,
                norm_correlation) = correlation_with_differences(data, uncertainty_type=uncertainty_type, use_sqrt=use_sqrt)
                # todo mittelwert ziehen bei vector correlations, norm correlations sollte skalar sein
                diff_vs_unc_correlations[uncertainty_type][use_sqrt] = np.mean(vector_correlations)
                dist_vs_unc_norm_correlations[uncertainty_type][use_sqrt] = norm_correlation

                if uncertainty_type == VAR_VEC and use_sqrt:
                    data_std_unc_norms = unc_norms
                if uncertainty_type == VAR_OF_VAR and use_sqrt:
                    dis_std_unc_norms = unc_norms
                if uncertainty_type == VAR_OF_MEAN and use_sqrt:
                    model_std_unc_norms = unc_norms
                    model_std_vector_correlations = vector_correlations

        result["diff_vs_unc_correlations"] = diff_vs_unc_correlations
        result["dist_vs_unc_norm_correlations"] = dist_vs_unc_norm_correlations

        print("4")
        unc_vs_unc_correlations = [ # 4
            np.corrcoef([data_std_unc_norms, dis_std_unc_norms])[0, 1],
            np.corrcoef([data_std_unc_norms, model_std_unc_norms])[0, 1],
            np.corrcoef([model_std_unc_norms, dis_std_unc_norms])[0, 1]
        ]
        result["unc_vs_unc_correlations"] = unc_vs_unc_correlations

        print("5")
        unc_vs_prec_correlations = {} # 5
        for unc in [VAR_VEC, VAR_OF_MEAN, VAR_OF_VAR]:
            unc_vs_prec_correlations[uncertainty_type] = {}
            for use_sqrt in [False, True]:
                unc_vs_prec_correlations[uncertainty_type][use_sqrt] = uncertainty_vs_precision(data, unc, use_sqrt, save_instead_of_show=False)
        result["unc_vs_prec_correlations"] = unc_vs_prec_correlations

        result["dist_vs_model_corrs_correlation"] = np.corrcoef([cos_distances, model_std_vector_correlations])[0, 1] # 6

        return result   

    relevant_paths = [p for p in RAW_DATA_PATHS]# if p not in [RAW_DATA_PATHS[0], RAW_DATA_PATHS[10]]] # exclude buggy runs with NaN values
    if not os.path.isfile("results.json"):
        results = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = None
            prev_path = None
            for path in relevant_paths: 
                next_future = executor.submit(load_data, path) # load data while previous data is being processed
                #data = load_data(path)
                if future is not None:
                    # Wait for the previous load to complete and process the data
                    data = future.result()
                    results[prev_path] = process_data(data)
                future = next_future
                prev_path = path
            if future is not None:
                data = future.result()
                results[prev_path] = process_data(data)
        
        print("-"*20)
        print(json.dumps(results))
        print("-"*20)
        with open("results.json", "w") as f:
            json.dump(results, f)
    else:
        print("loading results.json...")
        def bool_keys_hook(d):
            new_d = {}
            for k, v in d.items():
                if k == 'true':
                    new_d[True] = v
                elif k == 'false':
                    new_d[False] = v
                else:
                    new_d[k] = v
            return new_d
        with open("results.json", "r") as f:
            results = json.load(f, object_hook=bool_keys_hook)
            #print(json.dumps(results))
    
    baseline_path = "/usr/scratch4/angel8547/results/UAL/28/raw_model_outputs.json"

    def get_paths(x):
        paths = []
        if type(x) == dict:
            for k, v in x.items():
                v_paths = get_paths(v)
                for vp in v_paths:
                    if type(k) == str:
                        paths.append(f"['{k}']{vp}")
                    else:
                        paths.append(f"[{k}]{vp}")
                if len(v_paths) == 0:
                    if type(k) == str:
                        paths.append(f"['{k}']")
                    else: 
                        paths.append(f"[{k}]")
        elif type(x) == list:
            for i, x in enumerate(x):
                x_paths = get_paths(x)
                for xp in x_paths:
                    paths.append(f"[{i}]{xp}")
                if len(x_paths) == 0:
                    paths.append(f"[{i}]")
        else: # float
            pass
        return paths
    
    print("evaluation")

    summary = []
    print(1)
    summary2 = []
    print(2)
    paths = get_paths(results[baseline_path])
    print(3)
    #print(paths)
    for pth in paths:
        print(f"exec(results[{baseline_path}]{pth})")
        baseline_value = eval(f"results[baseline_path]{pth}")
        #print(baseline_value)
        #print(results['/usr/scratch4/angel8547/results/UAL/28/raw_model_outputs.json']['fv_vs_unc_correlations']['variance_vector']['Q'][True])
        all_values = []
        for path in relevant_paths:
            all_values.append(eval(f"results[path]{pth}"))

        #print(pth, baseline_value, all_values)

        most_different_value = max(all_values, key=lambda x:abs(x-baseline_value))

        summary.append(f"${baseline_value}$ & ${most_different_value}$ & ${abs(baseline_value-most_different_value)}$ & {pth} \\\\".replace('_', '\\_'))

        all_values = np.array(all_values)
        summary2.append(f"${np.mean(all_values)} \pm {np.std(all_values)}$ & {pth} \\\\".replace('_', '\\_'))

    print("\n\n")

    for summ in summary:
        print(summ)

    print("\n\n------------------------------\n\n")

    for summ in summary2:
        print(summ)

#verify_markus()

def check_raw_data_for_NaNs():
    """
    there are two runs containing inf/nan values: 10, 20
    10: 
        - seems like mostly 656 out of 2048 values are such per vector
        - 0.85% of total values are such
    20: 
        - seems like mostly 512 or 508 of 2048 values are such per vector
        - 0.66% of total values are such
    """
    for path in [ RAW_DATA_PATHS[16], RAW_DATA_PATHS[0], RAW_DATA_PATHS[10]]: # indices are wrong now, nan runs have been excluded

        data = np.array([vec  for vecs in load_data(path)["data"].values() for vec in vecs.values()])

        #for vec in data:
        #    vec[~np.isfinite(vec)] = 0.0
        
        all_finite = np.isfinite(data).all()
        print("All finite: ", all_finite)
        if not all_finite:
            print("num not finite: ", sum(np.ravel(~np.isfinite(data))) / len(np.ravel(data)))


#check_raw_data_for_NaNs()
            

def generate_uncertainty_score_distribution_plots():
    data = load_data()

    #plt.rc('text', usetex=True)

    for uncertainty_type in [VAR_VEC, VAR_OF_VAR, VAR_OF_MEAN, "fDATA", "fDIST", "fDATA+"]:

        fig = plt.figure()
        uncertainty_type_name = {
            VAR_VEC: "data_uncertainty",
            VAR_OF_VAR: "distributional_uncertainty",
            VAR_OF_MEAN: "model_uncertainty",
            "fDATA": "data_unc_filtered",
            "fDATA+": "data_unc_filtered_large",
            "fDIST": "dist_unc_filtered"
        }[uncertainty_type]
        filename = f"unc_score_distribution__{uncertainty_type_name}.pdf"

        x_axis_min, x_axis_max  = { # for entropy
            VAR_VEC: (2650, 2815),
            VAR_OF_VAR: (-6900, -4000),
            VAR_OF_MEAN: (-3500, -1300),
            "fDATA": (2650, 2815),
            "fDATA+": (2650, 2815),
            "fDIST": (-6900, -4500)
        }[uncertainty_type]
        # x_axis_min, x_axis_max  = { # for l2 norm
        #     VAR_VEC: (35.5, 41.5),
        #     VAR_OF_VAR: (0.004, 0.042),
        #     VAR_OF_MEAN: (0.0, 3.25),
        #     "fDATA": (35.5, 41.5),
        #     "fDATA+": (35.5, 41.5),
        #     "fDIST": (0.004, 0.042)
        # }[uncertainty_type]
        """x_axis_min, x_axis_max  = { # when using mean instead of norm
            VAR_VEC: (0.75, 0.9),
            VAR_OF_VAR: (0.0, 0.001),
            VAR_OF_MEAN: (0.0, 0.03)
        }[uncertainty_type]"""

        all_norms = {}
        norm_min_min = float("inf")
        norm_max_max = float("-inf")
        for set_id in ["Q", "D1", "D2", "D3", "D4"]:

            vectors = get_vectors(data, set_id, uncertainty_type)
            print(uncertainty_type_name, set_id, len(vectors), f"={100* len(vectors) / NUM_SAMPLES_IN_SET[set_id]:.1f}% at {NUM_STDS_FOR_CUTOFF} sigma")
            #norms = np.linalg.norm(vectors, 2, 1)
            norms = 1024 * np.log(2 * np.pi * np.e) + 0.5 * np.sum(np.log(vectors), axis=1)
            #norms = np.mean(vectors, axis=1)
            all_norms[set_id] = norms

            norm_min = np.min(norms)
            norm_max = np.max(norms)

            #x_axis_max = norm_max
            #x_axis_min = norm_min

            if norm_min < norm_min_min:
                norm_min_min = norm_min
            if norm_max > norm_max_max:
                norm_max_max = norm_max

        x = np.linspace(x_axis_min, x_axis_max, 300)
        
        max_max_y = 0
        for set_id in ["Q", "D1", "D2", "D3", "D4"]:


            set_color = {
                "Q": "k",
                "D1": "b",
                "D2": "g",
                "D3": "y",
                "D4": "r"
            }[set_id]

            norms = all_norms[set_id]

            if uncertainty_type == "fDATA" and set_id == "D4":
                print(f"{norms=}")
                # plt.plot([norms[0], norms[0]], [max_y_Q*0.97, max_y_Q*1.01], "-", color=set_color, linewidth=2, label=set_id)
                # continue # 1 sample (@ 1.5 sig) is not enough for meaningful KDE

            # generate KDE plot 
            norm_min = np.min(norms)
            norm_max = np.max(norms)
            norm_range_size = norm_max - norm_min

            if uncertainty_type in ["fDATA", "fDIST", "fDATA+"]:
                bandwidth=(norm_range_size*0.05)
            else:
                bandwidth=(norm_range_size*0.03)

            if uncertainty_type == "fDATA" and set_id == "D4" and False:
                bandwidth = (0.02)

            kde = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
            kde.fit(np.array(norms)[:, np.newaxis])
            
            y = np.exp(kde.score_samples(x[:, np.newaxis]))

            if uncertainty_type == "fDATA" and set_id == "D4":
                # y = y*0.06 # scale to not destroy the plot
                y = y*0.15# scale to not destroy the plot

            max_y = np.max(y)
            if max_y > max_max_y:
                max_max_y = max_y

            mean_norm = np.mean(norms)
            norm_std = np.std(norms)
                
            plt.plot(x, y, "-", color=set_color, label=set_id)
            plt.fill_between(x, y, color=set_color, alpha=0.3)
            #plt.plot([x_mean, x_mean], [0, np.max(y) * 0.05], 'k-')
            std_axis_y_factor = 1.08
            if set_id == "Q":
                max_y_Q = max_y
                std_axis_y = max_y*std_axis_y_factor
                
                plt.plot([mean_norm, mean_norm], [max_y*(std_axis_y_factor - 0.025), max_y*(std_axis_y_factor + 0.025)], "-", color=set_color, linewidth=2)
                print(f"Query: {uncertainty_type_name}, {mean_norm=}, {norm_std=}")
                text_y = max_y*(std_axis_y_factor + 0.03)
                plt.text(mean_norm, max_y * (std_axis_y_factor + 0.04), r"$\mu$", horizontalalignment="center", fontsize=11)
                
                i = -1
                while mean_norm + i*norm_std < x_axis_max:
                    plt.plot([mean_norm + i*norm_std, mean_norm + i*norm_std], [0, max_y*(std_axis_y_factor - 0.015)], ":", color="grey", alpha=0.75, linewidth=0.5)
                    if i == 0 or (i == 17 and uncertainty_type == "fDIST"): 
                        i += 1
                        continue
                    plt.plot([mean_norm + i*norm_std, mean_norm + i*norm_std], [max_y*(std_axis_y_factor - 0.015), max_y*(std_axis_y_factor + 0.015)], "-", color=set_color, linewidth=0.5)
                    plt.text(mean_norm + i*norm_std, text_y, r"$" + str(i) + r"\sigma$", horizontalalignment="center", fontsize=11)
                    i += 1
                i -= 1

                
                plt.text(mean_norm - norm_std*1.2, std_axis_y, "means:  ", horizontalalignment="right", verticalalignment="center", fontsize=11)

                plt.plot([mean_norm - norm_std, mean_norm + i*norm_std], [std_axis_y, std_axis_y], "-", color=set_color, linewidth=0.5)
            else:

                offset = {
                    "D1": 0.015,
                    "D2": 0.03,
                    "D3": 0.045,
                    "D4": 0.06
                }[set_id]
                #mean
                plt.plot([mean_norm, mean_norm], [max_y_Q*(std_axis_y_factor-0.02-offset), max_y_Q*(std_axis_y_factor+0.02-offset)], "-", color=set_color, linewidth=2)
                #std vertical
                plt.plot([mean_norm - norm_std, mean_norm - norm_std], [max_y_Q*(std_axis_y_factor-0.01-offset), max_y_Q*(std_axis_y_factor+0.01-offset)], "-", color=set_color, linewidth=0.25)
                plt.plot([mean_norm + norm_std, mean_norm + norm_std], [max_y_Q*(std_axis_y_factor-0.01-offset), max_y_Q*(std_axis_y_factor+0.01-offset)], "-", color=set_color, linewidth=0.25)
                #std horizontal
                plt.plot([mean_norm - norm_std, mean_norm + norm_std], [max_y_Q*(std_axis_y_factor-offset), max_y_Q*(std_axis_y_factor-offset)], "-", color=set_color, linewidth=0.1)


        x_label = {
            VAR_VEC: r"Data Uncertainty Score $\left\| \Sigma^{(D)} \right\|$",
            VAR_OF_VAR: r"Distributional Uncertainty Score $\left\| \Sigma^{(V)} \right\|$",
            VAR_OF_MEAN: r"Model Uncertainty Score $\left\| \Sigma^{(M)} \right\|$",
            "fDATA": r'Data Uncertainty Score $\left\| \Sigma^{(D)} \right\|$',
            "fDATA+": r'Data Uncertainty Score $\left\| \Sigma^{(D)} \right\|$',
            "fDIST": r"Distributional Uncertainty Score $\left\| \Sigma^{(V)} \right\|$"
        }[uncertainty_type]
        x_label_bbox = {
            VAR_VEC: bbox_dat,
            VAR_OF_VAR: bbox_dis,
            VAR_OF_MEAN: bbox_mod,
            "fDATA": bbox_dat,
            "fDATA+": bbox_dat,
            "fDIST": bbox_dis
        }[uncertainty_type]
        # x_label = {
        #     VAR_VEC: r"$L_2$-Norm of Data Uncertainty Vector",
        #     VAR_OF_VAR: r"$L_2$-Norm of Distributional Uncertainty Vector",
        #     VAR_OF_MEAN: r"$L_2$-Norm of Model Uncertainty Vector",
        #     "fDATA": r'$L_2$-Norm of Data Uncertainty Vector',
        #     "fDIST": r"$L_2$-Norm of Distributional Uncertainty Vector"
        # }[uncertainty_type]

        plt.xlabel(x_label, fontsize=20, bbox=x_label_bbox, labelpad=7) 
        plt.ylabel(r"Probability Density", fontsize=15)

        #plt.title(title)
        if uncertainty_type == "fDATA":
            plt.title(r"Considering images where $\| \Sigma^{(V)} \| <" + DIST_CUTOFF_STR + r"$", fontsize=15)
        elif uncertainty_type == "fDIST":
            plt.title(r"Considering images where $\| \Sigma^{(D)} \| <" + DATA_CUTOFF_STR + r"$", fontsize=15)
        elif uncertainty_type == "fDATA+":
            plt.title(r"Considering images where $\| \Sigma^{(V)} \| >" + DIST_CUTOFF_LARGE_STR + r"$", fontsize=15)

        plt.axis([x_axis_min, x_axis_max, 0, max_max_y * 1.16])
        
        leg = plt.legend(loc="center right", fontsize=15)
        # set the linewidth of each legend object
        for legobj in leg.legend_handles:
            legobj.set_linewidth(4.0)

        plt.tight_layout()
        plt.savefig(filename, dpi=600, transparent=True)
        plt.close(fig)



def get_distractor_subset_examples():
    data = load_data()
    for set_id, filenames in data["sets"].items():
        examples_names = random.sample(filenames, 10)

        os.mkdir(set_id)

        if set_id == "Q":
            images_path = QUERY_PATH
        else:
            images_path = GALLERY_PATH

        for name in examples_names:
            shutil.copyfile(os.path.join(images_path, name), os.path.join(set_id, name))
        
#get_distractor_subset_examples()
            


def test_memory_available():
    import psutil
    all_data = []
    for path in RAW_DATA_PATHS:
        print(path)
        print(f"available memory: {psutil.virtual_memory().available / (1024 ** 3):.2f} GB")
        data = load_data(path)
        all_data.append(data)
    
    print(f"success! {len(all_data)=}")

#test_memory_available()


#----------------------------------------------------------------------------------------------------
def get_vector_type(name):
    """translates model/data/dist to key needed for get_vectors"""
    return {
        "model": VAR_OF_MEAN,
        "data": VAR_VEC,
        "dist": VAR_OF_VAR
    }[name]

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

def get_stats_container(float_list):
    """upgrade your list to a dict containing the list as well as some statistics about it"""
    return {
        "min": np.min(float_list),
        "mean": np.mean(float_list),
        "std": np.std(float_list),
        "max": np.max(float_list),
        "all": float_list
    }

def get_stats_and_delta_container(float_list, delta_list):
    """"""
    stats_container = get_stats_container(float_list)
    stats_container["deltas"] = get_stats_container(delta_list)

    return stats_container

def get_QG_vecs(data, vector_type):
    """wrapper for getting query and gallery vectors to avoid boilerplate"""
    return torch.Tensor(get_vectors(data, "Q", vector_type)), torch.Tensor(get_vectors(data, "G", vector_type))
    #return torch.Tensor(get_vectors(data, "Q", vector_type)).cuda(), torch.Tensor(get_vectors(data, "G", vector_type)).cuda()

def ensure_key_exists(dict_, key):
    #print(dict_)
    if key not in dict_.keys():
        dict_[key] = {}


def get_vanilla_performances(all_data):
    """calculate mAP and rank1 for all given runs
    
    returns mAP, rank1 (each generated by `get_stats_container`)
    """

    mAPs: list[float] = []
    rank1s: list[float] = []

    for data in all_data:
        query_vecs, gallery_vecs = get_QG_vecs(data, MEAN_VEC) 

        dist_mat = compute_cosine_distance(query_vecs, gallery_vecs)

        mAP = get_mAP(dist_mat, data)
        mAPs.append(mAP)

        rank1 = get_rank1(dist_mat, data)
        rank1s.append(rank1)

    mAP_deltas: list[float] = [ 0.0 for _ in mAPs] # there is no difference because this is the baseline
    rank1_deltas: list[float] = [ 0.0 for _ in rank1s]

    return get_stats_and_delta_container(mAPs, mAP_deltas), get_stats_and_delta_container(rank1s, rank1_deltas)



def get_sigma_as_fv_performances(all_data, augmentation_basis, augmentation_basis_is_std, base_mAPs, base_rank1s):
    """calculate mAP and rank1 for all given runs when using a given uncertainty vector as the feature vector

    args:
        - `all_data`: list of data objects (see `load_data`)
        - `augmentation_basis`: name of uncertainty type (model/data/dist)
        - `augmentation_basis_is_std`: whether to take elementwise sqrt of uncertainty vectors before using them
    
    returns mAP, rank1 (each generated by `get_stats_container`)
    """
    mAPs: list[float] = []
    rank1s: list[float] = []

    for data in all_data:
        query_vecs, gallery_vecs = get_QG_vecs(data, get_vector_type(augmentation_basis))
        
        if augmentation_basis_is_std:
            query_vecs = torch.sqrt(query_vecs)
            gallery_vecs = torch.sqrt(gallery_vecs)

        dist_mat = compute_cosine_distance(query_vecs, gallery_vecs)

        mAP = get_mAP(dist_mat, data)
        mAPs.append(mAP)

        rank1 = get_rank1(dist_mat, data)
        rank1s.append(rank1)

    mAP_deltas: list[float] = [ b - a for a, b in zip(base_mAPs, mAPs)]
    rank1_deltas: list[float] = [ b - a for a, b in zip(base_rank1s, rank1s)]

    return get_stats_and_delta_container(mAPs, mAP_deltas), get_stats_and_delta_container(rank1s, rank1_deltas)

def get_add_distmat_aug_performances(all_data, augmentation_basis, augmentation_basis_is_std, base_mAPs, base_rank1s):
    """calculate mAP and rank1 for all given runs when adding the distmat obtained by interpreting the 
    uncertainty vector as a feature vector to the normal distmat (based on actual FVs)

    args:
        - `all_data`: list of data objects (see `load_data`)
        - `augmentation_basis`: name of uncertainty type (model/data/dist)
        - `augmentation_basis_is_std`: whether to take elementwise sqrt of uncertainty vectors before using them
    
    returns mAP, rank1 (each generated by `get_stats_container`)
    """
    mAPs: list[float] = []
    rank1s: list[float] = []

    for data in all_data:
        query_unc_vecs, gallery_unc_vecs = get_QG_vecs(data, get_vector_type(augmentation_basis))
        query_fvs, gallery_fvs = get_QG_vecs(data, MEAN_VEC)
        
        if augmentation_basis_is_std:
            query_unc_vecs = torch.sqrt(query_unc_vecs)
            gallery_unc_vecs = torch.sqrt(gallery_unc_vecs)

        unc_dist_mat = compute_cosine_distance(query_unc_vecs, gallery_unc_vecs)
        fv_dist_mat = compute_cosine_distance(query_fvs, gallery_fvs)

        dist_mat = fv_dist_mat + unc_dist_mat

        mAP = get_mAP(dist_mat, data)
        mAPs.append(mAP)

        rank1 = get_rank1(dist_mat, data)
        rank1s.append(rank1)

    mAP_deltas: list[float] = [ b - a for a, b in zip(base_mAPs, mAPs)]
    rank1_deltas: list[float] = [ b - a for a, b in zip(base_rank1s, rank1s)]

    return get_stats_and_delta_container(mAPs, mAP_deltas), get_stats_and_delta_container(rank1s, rank1_deltas)

def get_mul_distmat_aug_performances(all_data, augmentation_basis, augmentation_basis_is_std, base_mAPs, base_rank1s):
    """calculate mAP and rank1 for all given runs when calculating the distmat as $ D = 1 - ( 1 - D_{FV} ) \odot ( 1 - D_{unc} ) $
    where $ D_{FV} $ is the distance matrix based on the feature vector and $ D_{unc} $ is the distance matrix based on the uncertainty vector.

    args:
        - `all_data`: list of data objects (see `load_data`)
        - `augmentation_basis`: name of uncertainty type (model/data/dist)
        - `augmentation_basis_is_std`: whether to take elementwise sqrt of uncertainty vectors before using them
    
    returns mAP, rank1 (each generated by `get_stats_container`)
    """
    mAPs: list[float] = []
    rank1s: list[float] = []

    for data in all_data:
        query_unc_vecs, gallery_unc_vecs = get_QG_vecs(data, get_vector_type(augmentation_basis))
        query_fvs, gallery_fvs = get_QG_vecs(data, MEAN_VEC)
        
        if augmentation_basis_is_std:
            query_unc_vecs = torch.sqrt(query_unc_vecs)
            gallery_unc_vecs = torch.sqrt(gallery_unc_vecs)

        unc_dist_mat = compute_cosine_distance(query_unc_vecs, gallery_unc_vecs)
        fv_dist_mat = compute_cosine_distance(query_fvs, gallery_fvs)

        dist_mat = 1 - ( 1 - fv_dist_mat ) * ( 1 - unc_dist_mat )

        mAP = get_mAP(dist_mat, data)
        mAPs.append(mAP)

        rank1 = get_rank1(dist_mat, data)
        rank1s.append(rank1)

    mAP_deltas: list[float] = [ b - a for a, b in zip(base_mAPs, mAPs)]
    rank1_deltas: list[float] = [ b - a for a, b in zip(base_rank1s, rank1s)]

    return get_stats_and_delta_container(mAPs, mAP_deltas), get_stats_and_delta_container(rank1s, rank1_deltas)

def get_epsilon_weighting_performances(all_data, augmentation_basis, augmentation_basis_is_std, maximization_goal, base_mAPs, base_rank1s):
    """calculate mAP and rank1 for all given runs when weighing the feature vector with 1/(uncertainty + eps)
    where eps is a hyperparameter that is chosen by applying `maximization_goal` to a hyperparameter search
    over all runs.

    args:
        - `all_data`: list of data objects (see `load_data`)
        - `augmentation_basis`: name of uncertainty type (model/data/dist)
        - `augmentation_basis_is_std`: whether to take elementwise sqrt of uncertainty vectors before using them
        - `maximization goal`: how to choose the eps that wins the hyperparameter search over all runs
        - `base_mAPs`: the vanilla mAPs of the runs
        - `base_rank1s`: the vanilla rank1s of the runs
    
    returns mAP, rank1 (each generated by `get_stats_container`)
    """
    
    eps_mAP_map = {}
    eps_rank1_map = {}

    for data in tqdm(all_data, "Hyperparameter Search"):

        query_unc_vecs, gallery_unc_vecs = get_QG_vecs(data, get_vector_type(augmentation_basis))
        query_fvs, gallery_fvs = get_QG_vecs(data, MEAN_VEC)
        
        if augmentation_basis_is_std:
            query_unc_vecs = np.sqrt(query_unc_vecs)
            gallery_unc_vecs = np.sqrt(gallery_unc_vecs)
            # query_unc_vecs = torch.sqrt(query_unc_vecs)
            # gallery_unc_vecs = torch.sqrt(gallery_unc_vecs)

        # hyperparameter search 
        hint_value = 0.2438
        delta = 0.0001
        factor = 2 ** (1/4)    # 4th root of 2
        iterations = int(np.log(hint_value/delta)/np.log(factor))
        sub = factor ** 10 - 1
        diff = delta * (factor ** np.arange(10, iterations+1) - sub)
        epsilon_values = sorted([0, hint_value] + list(hint_value - diff) +
                                list(hint_value + diff))
        

        for epsilon in epsilon_values:

            if epsilon not in eps_mAP_map.keys():
                eps_mAP_map[epsilon] = []
            if epsilon not in eps_rank1_map.keys():
                eps_rank1_map[epsilon] = []

            weighted_query_fvs = torch.Tensor(query_fvs / (query_unc_vecs + epsilon))
            weighted_gallery_fvs = torch.Tensor(gallery_fvs / (gallery_unc_vecs + epsilon))
            # weighted_query_fvs = query_fvs / (query_unc_vecs + epsilon)
            # weighted_gallery_fvs = gallery_fvs / (gallery_unc_vecs + epsilon)

            dist_mat = compute_cosine_distance(weighted_query_fvs, weighted_gallery_fvs)
            
            mAP = get_mAP(dist_mat, data)
            print(f"for {epsilon=} {mAP=}")
            eps_mAP_map[epsilon].append(mAP)

            rank1 = get_rank1(dist_mat, data) 
            eps_rank1_map[epsilon].append(rank1) 
            

    # choose best eps based on maximazation goal
    best_eps = 0.0
    best_avg_delta = float("-inf")
    best_avg_mAP = float("-inf")

    for eps, mAP_list in eps_mAP_map.items(): # if we use epsilon=eps, the mAPs of the 20 runs are in mAP_list
        
        avg_mAP = np.mean(mAP_list)
        deltas = [ b - a for a, b in zip(base_mAPs, mAP_list)]
        avg_delta = np.mean(deltas)
        
        # choose condition that decides if eps is better than best_eps based on maximization_goal
        eps_is_better_than_best_eps = { 
            "best_avg_delta-mAP": avg_delta > best_avg_delta, 
            "best_avg_res-mAP": avg_mAP > best_avg_mAP
        }[maximization_goal]

        if eps_is_better_than_best_eps:
            best_eps = eps 
            best_avg_delta = avg_delta
            best_avg_mAP = avg_mAP # not strictly correct to update both at the same time like this but it doesn't matter for this usage
            # this is ok as long as we do not use the one that the maximization strategy doesn't care about # we don't use either so it's fine

    # return mAPs and rank1s for that best eps
    return_mAPs: list[float] = eps_mAP_map[best_eps]
    return_rank1s: list[float] = eps_rank1_map[best_eps]
    return_mAP_deltas: list[float] = [ b - a for a, b in zip(base_mAPs, return_mAPs)]
    return_rank1_deltas: list[float] = [ b - a for a, b in zip(base_rank1s, return_rank1s)] 

    mAP_container, rank1_container = get_stats_and_delta_container(return_mAPs, return_mAP_deltas), get_stats_and_delta_container(return_rank1s, return_rank1_deltas)

    mAP_container["eps"] = best_eps
    rank1_container["eps"] = best_eps

    print(f"       best epsilon: {best_eps}")

    return mAP_container, rank1_container

def get_lambda_weighting_performances(all_data, augmentation_basis, augmentation_basis_is_std, maximization_goal, augmentation_auxiliary, augmentation_auxiliary_is_std, base_mAPs, base_rank1s, norm_func=lambda x: torch.norm(x, 2, 1, keepdim=True), hint_value = 0.2438):
    """calculate mAP and rank1 for all given runs when weighing the feature vector with 1/(uncertainty + eps)
    where eps = lambda \| uncertainty \| and lambda a hyperparameter that is chosen by applying `maximization_goal` to a hyperparameter search
    over all runs.

    args:
        - `all_data`: list of data objects (see `load_data`)
        - `augmentation_basis`: name of uncertainty type (model/data/dist)
        - `augmentation_basis_is_std`: whether to take elementwise sqrt of uncertainty vectors before using them
        - `maximization goal`: how to choose the eps that wins the hyperparameter search over all runs
        - `base_mAPs`: the vanilla mAPs of the runs
        - `base_rank1s`: the vanilla rank1s of the runs
    
    returns mAP, rank1 (each generated by `get_stats_container`)
    """

    
    delta = 0.0001 * hint_value / 0.2438 # use same setup as other hyperparam searches but scale delta according to change in hint value
    # delta = np.abs(0.0001 * hint_value / 0.2438) # use same setup as other hyperparam searches but scale delta according to change in hint value
    #if delta < 1e-6: delta = 1e-6


    lambda_mAP_map = {}
    lambda_rank1_map = {}

    for data in tqdm(all_data, "Hyperparameter Search"):

        query_unc_vecs, gallery_unc_vecs = get_QG_vecs(data, get_vector_type(augmentation_basis))
        query_aux_unc_vecs, gallery_aux_unc_vecs = get_QG_vecs(data, get_vector_type(augmentation_auxiliary))
        query_fvs, gallery_fvs = get_QG_vecs(data, MEAN_VEC)
        
        if augmentation_basis_is_std:
            query_unc_vecs = torch.sqrt(query_unc_vecs)
            gallery_unc_vecs = torch.sqrt(gallery_unc_vecs)
        if augmentation_auxiliary_is_std:
            query_aux_unc_vecs = torch.sqrt(query_aux_unc_vecs)
            gallery_aux_unc_vecs = torch.sqrt(gallery_aux_unc_vecs)

        query_aux_unc_vec_norms = norm_func(query_aux_unc_vecs) # keepdim for broadcasting later
        gallery_aux_unc_vec_norms = norm_func(gallery_aux_unc_vecs)
        # query_aux_unc_vec_norms = torch.norm(query_aux_unc_vecs, 2, 1, keepdim=True) # keepdim for broadcasting later
        # gallery_aux_unc_vec_norms = torch.norm(gallery_aux_unc_vecs, 2, 1, keepdim=True)

        # hyperparameter search 
        
        
        factor = 2 ** (1/4)    # 4th root of 2
        iterations = int(np.log(hint_value/delta)/np.log(factor))
        sub = factor ** 10 - 1
        diff = delta * (factor ** np.arange(10, iterations+1) - sub)
        lambda_values = sorted([0, hint_value] + list(hint_value - diff) +
                                list(hint_value + diff))
        # 2905.9861160031696 + 0.5 * torch.sum(torch.log(query_aux_unc_vecs), dim=1, keepdim=True)
        #print(f"{lambda_values=}")
        for lambda_ in lambda_values:

            if lambda_ not in lambda_mAP_map.keys():
                lambda_mAP_map[lambda_] = []
            if lambda_ not in lambda_rank1_map.keys():
                lambda_rank1_map[lambda_] = []

            query_epsilon = lambda_ * query_aux_unc_vec_norms
            #print(f"computed epsilon in mean: {torch.mean(query_epsilon)}")
            gallery_epsilon = lambda_ * gallery_aux_unc_vec_norms

            weighted_query_fvs = query_fvs / (query_unc_vecs + query_epsilon)
            weighted_gallery_fvs = gallery_fvs / (gallery_unc_vecs + gallery_epsilon)

            dist_mat = compute_cosine_distance(weighted_query_fvs, weighted_gallery_fvs)
            
            mAP = get_mAP(dist_mat, data)
            lambda_mAP_map[lambda_].append(mAP)

            rank1 = get_rank1(dist_mat, data) 
            lambda_rank1_map[lambda_].append(rank1) 
            
    #print(f"{lambda_mAP_map=}")
    # choose best lambda based on maximazation goal
    best_lambda = 0.0
    best_avg_delta = float("-inf")
    best_avg_mAP = float("-inf")

    for lambda_, mAP_list in lambda_mAP_map.items(): # if we use lambda=lambda_, the mAPs of the 20 runs are in mAP_list
        
        avg_mAP = np.mean(mAP_list)
        deltas = [ b - a for a, b in zip(base_mAPs, mAP_list)]
        avg_delta = np.mean(deltas)
        
        # choose condition that decides if lambda is better than best_lambda based on maximization_goal
        lambda_is_better_than_best_lambda = { 
            "best_avg_delta-mAP": avg_delta > best_avg_delta, 
            "best_avg_res-mAP": avg_mAP > best_avg_mAP
        }[maximization_goal]

        if lambda_is_better_than_best_lambda:
            best_lambda = lambda_
            best_avg_delta = avg_delta
            best_avg_mAP = avg_mAP # not strictly correct to update both at the same time like this but it doesn't matter for this usage
            # this is ok as long as we do not use the one that the maximization strategy doesn't care about # we don't use either so it's fine

    # return mAPs and rank1s for that best eps
    return_mAPs: list[float] = lambda_mAP_map[best_lambda]
    return_rank1s: list[float] = lambda_rank1_map[best_lambda]
    return_mAP_deltas: list[float] = [ b - a for a, b in zip(base_mAPs, return_mAPs)]
    return_rank1_deltas: list[float] = [ b - a for a, b in zip(base_rank1s, return_rank1s)] 

    mAP_container, rank1_container = get_stats_and_delta_container(return_mAPs, return_mAP_deltas), get_stats_and_delta_container(return_rank1s, return_rank1_deltas)

    mAP_container["best_lambda"] = best_lambda
    mAP_container["lambda_mAP_map"] = lambda_mAP_map

    print(f"       best lambda: {best_lambda}")

    return mAP_container, rank1_container

def get_nomagic_weighting_aug_performances(all_data, augmentation_basis, augmentation_basis_is_std, base_mAPs, base_rank1s):
    """calculate mAP and rank1 for all given runs when weighing the feature vector with 1/(uncertainty + eps)
    where eps is calculated as `2048 / sum(abs(log(unc_vecs)))`.

    args:
        - `all_data`: list of data objects (see `load_data`)
        - `augmentation_basis`: name of uncertainty type (model/data/dist)
        - `augmentation_basis_is_std`: whether to take elementwise sqrt of uncertainty vectors before using them
        - `base_mAPs`: the vanilla mAPs of the runs
        - `base_rank1s`: the vanilla rank1s of the runs
    
    returns mAP, rank1 (each generated by `get_stats_and_delta_container`)
    """
    
    mAPs: list[float] = []
    rank1s: list[float] = []

    for data in all_data:
        query_unc_vecs, gallery_unc_vecs = get_QG_vecs(data, get_vector_type(augmentation_basis))
        query_fvs, gallery_fvs = get_QG_vecs(data, MEAN_VEC)
        
        if augmentation_basis_is_std:
            query_unc_vecs = torch.sqrt(query_unc_vecs)
            gallery_unc_vecs = torch.sqrt(gallery_unc_vecs)

        epsQ = 1024 / torch.sum(torch.abs(torch.log(query_unc_vecs)), axis=1, keepdim=True) 
        epsG = 1024 / torch.sum(torch.abs(torch.log(gallery_unc_vecs)), axis=1, keepdim=True) 

        weighted_query_fvs = torch.Tensor(query_fvs / (query_unc_vecs + epsQ))
        weighted_gallery_fvs = torch.Tensor(gallery_fvs / (gallery_unc_vecs + epsG))
        # weighted_query_fvs = query_fvs / (query_unc_vecs + epsilon)
        # weighted_gallery_fvs = gallery_fvs / (gallery_unc_vecs + epsilon)

        dist_mat = compute_cosine_distance(weighted_query_fvs, weighted_gallery_fvs)

        mAP = get_mAP(dist_mat, data)
        mAPs.append(mAP)

        rank1 = get_rank1(dist_mat, data)
        rank1s.append(rank1)

    mAP_deltas: list[float] = [ b - a for a, b in zip(base_mAPs, mAPs)]
    rank1_deltas: list[float] = [ b - a for a, b in zip(base_rank1s, rank1s)]

    return get_stats_and_delta_container(mAPs, mAP_deltas), get_stats_and_delta_container(rank1s, rank1_deltas)


def get_all_performances(paths, augmentation_bases, out_path):
    """calculate mAP and rank1 for all UAL uncertainty usage experiments and runs"""

    print("loading all data...")
    all_data = []
    # ["/data2/hpcuser/angel8547/results_backup_2024_03_13/UAL/28/raw_model_outputs.json"]:#
    # [f"/data2/hpcuser/angel8547/results_backup_2024_03_13/UAL/{i}/raw_model_outputs.json" for i in [j for j in range(11,20)] + [j for j in range(21, 30)]]:
    # RAW_DATA_PATHS
    for path in paths:
        data = load_data(path)
        all_data.append(data)

    with open("hint_values.json", "r") as f:
        def bool_keys_hook(d):
            new_d = {}
            for k, v in d.items():
                if k == 'true':
                    new_d[True] = v
                elif k == 'false':
                    new_d[False] = v
                else:
                    new_d[k] = v
            return new_d
        unc_hint_map = json.load(f, object_hook=bool_keys_hook)
    mAPs = {}
    rank1s = {}

    print(f"computing vanilla performance...")
    vanilla_mAPs, vanilla_rank1s = get_vanilla_performances(all_data)
    mAPs["vanilla"] = vanilla_mAPs
    rank1s["vanilla"] = vanilla_rank1s

    ensure_key_exists(mAPs, "aug")
    ensure_key_exists(rank1s, "aug")

    for augmentation_basis in augmentation_bases: 
        print(f"- {augmentation_basis=}")
        ensure_key_exists(mAPs['aug'], augmentation_basis)
        ensure_key_exists(rank1s['aug'], augmentation_basis)

        for augmentation_basis_is_std in [True, False]:
            print(f"-- {augmentation_basis_is_std=}")
            ensure_key_exists(mAPs['aug'][augmentation_basis], augmentation_basis_is_std)
            ensure_key_exists(rank1s['aug'][augmentation_basis], augmentation_basis_is_std)

            # getter output is tuple: (mAPs, rank1s), each with {mean, std, max, all}
            """print(f"   computing performance for using the uncertainty vector as the feature vector...")

            sigma_as_fv_mAPs, sigma_as_fv_rank1s = get_sigma_as_fv_performances(
                all_data,
                augmentation_basis, 
                augmentation_basis_is_std, 
                mAPs["vanilla"]["all"], 
                rank1s["vanilla"]["all"] )
            
            mAPs["aug"][augmentation_basis][augmentation_basis_is_std]["sigma_as_fv"] = sigma_as_fv_mAPs
            rank1s["aug"][augmentation_basis][augmentation_basis_is_std]["sigma_as_fv"] = sigma_as_fv_rank1s"""

            
            """print(f"   computing performance for additively augmenting the distance matrix using the uncertainty distance matrix...")
            
            add_distmat_aug_mAPs, add_distmat_aug_rank1s = get_add_distmat_aug_performances(
                all_data, 
                augmentation_basis, 
                augmentation_basis_is_std, 
                mAPs["vanilla"]["all"], 
                rank1s["vanilla"]["all"] )
            
            mAPs["aug"][augmentation_basis][augmentation_basis_is_std]["add_distmat_aug"] = add_distmat_aug_mAPs
            rank1s["aug"][augmentation_basis][augmentation_basis_is_std]["add_distmat_aug"] = add_distmat_aug_rank1s"""


            """print(f"   computing performance for multiplicatively augmenting the distance matrix using the uncertainty distance matrix...")
            
            mul_distmat_aug_mAPs, mul_distmat_aug_rank1s = get_mul_distmat_aug_performances(
                all_data, 
                augmentation_basis, 
                augmentation_basis_is_std, 
                mAPs["vanilla"]["all"], 
                rank1s["vanilla"]["all"] )
            
            mAPs["aug"][augmentation_basis][augmentation_basis_is_std]["mul_distmat_aug"] = mul_distmat_aug_mAPs
            rank1s["aug"][augmentation_basis][augmentation_basis_is_std]["mul_distmat_aug"] = mul_distmat_aug_rank1s"""
            
            
            """print(f"   computing performance for fv weighting scheme with directly computed epsilon (nomagic)...")
            
            nomagic_weighting_aug_mAPs, nomagic_weighting_aug_rank1s = get_nomagic_weighting_aug_performances(
                all_data, 
                augmentation_basis, 
                augmentation_basis_is_std, 
                mAPs["vanilla"]["all"], 
                rank1s["vanilla"]["all"] )
            
            mAPs["aug"][augmentation_basis][augmentation_basis_is_std]["fv_weighting_nomagic"] = nomagic_weighting_aug_mAPs
            rank1s["aug"][augmentation_basis][augmentation_basis_is_std]["fv_weighting_nomagic"] = nomagic_weighting_aug_rank1s"""

            # weighting schemes
            ensure_key_exists(
                mAPs["aug"][augmentation_basis][augmentation_basis_is_std], 
                "fv_weighting" )
            ensure_key_exists(
                rank1s["aug"][augmentation_basis][augmentation_basis_is_std], 
                "fv_weighting" )
            
            for maximization_goal in ["best_avg_delta-mAP"]:#, "best_avg_res-mAP"]: # does not make a difference
                print(f"--- {maximization_goal=}")
                ensure_key_exists(
                    mAPs["aug"][augmentation_basis][augmentation_basis_is_std]["fv_weighting"], 
                    maximization_goal )
                ensure_key_exists(
                    rank1s["aug"][augmentation_basis][augmentation_basis_is_std]["fv_weighting"], 
                    maximization_goal )

                """print(f"    computing performance for using a constant c in the feature vector weighting scheme...")
                
                epsilon_weighting_mAPs, epsilon_weighting_rank1s = get_epsilon_weighting_performances(
                    all_data, 
                    augmentation_basis, 
                    augmentation_basis_is_std, 
                    maximization_goal, 
                    mAPs["vanilla"]["all"], 
                    rank1s["vanilla"]["all"] )
                
                mAPs["aug"][augmentation_basis][augmentation_basis_is_std]["fv_weighting"][maximization_goal]["eps"] = epsilon_weighting_mAPs
                rank1s["aug"][augmentation_basis][augmentation_basis_is_std]["fv_weighting"][maximization_goal]["eps"] = epsilon_weighting_rank1s"""

                #continue # disable for re-run
                ensure_key_exists(
                    mAPs["aug"][augmentation_basis][augmentation_basis_is_std]["fv_weighting"][maximization_goal], 
                    "lambda" )
                ensure_key_exists(
                    rank1s["aug"][augmentation_basis][augmentation_basis_is_std]["fv_weighting"][maximization_goal], 
                    "lambda" )
                
                for augmentation_auxiliary in ["model", "data", "dist"]:
                    ensure_key_exists(
                        mAPs["aug"][augmentation_basis][augmentation_basis_is_std]["fv_weighting"][maximization_goal]["lambda"], 
                        augmentation_auxiliary )
                    ensure_key_exists(
                        rank1s["aug"][augmentation_basis][augmentation_basis_is_std]["fv_weighting"][maximization_goal]["lambda"], 
                        augmentation_auxiliary )
                    print(f"---- {augmentation_auxiliary=}")

                    for augmentation_auxiliary_is_std in [True, False]:
                        print(f"----- {augmentation_auxiliary_is_std=}")
                        ensure_key_exists(
                            mAPs["aug"][augmentation_basis][augmentation_basis_is_std]["fv_weighting"][maximization_goal]["lambda"][augmentation_auxiliary], 
                            augmentation_auxiliary_is_std )
                        ensure_key_exists(
                            rank1s["aug"][augmentation_basis][augmentation_basis_is_std]["fv_weighting"][maximization_goal]["lambda"][augmentation_auxiliary], 
                            augmentation_auxiliary_is_std )

                        """print(f"      computing performance for using a derived c weighted by a constant lambda in the feature vector weighting scheme...")
                        
                        lambda_weighting_mAPs, lambda_weighting_rank1s = get_lambda_weighting_performances(
                            all_data, 
                            augmentation_basis, 
                            augmentation_basis_is_std, 
                            maximization_goal,
                            augmentation_auxiliary, 
                            augmentation_auxiliary_is_std, 
                            mAPs["vanilla"]["all"], 
                            rank1s["vanilla"]["all"],
                            lambda x: torch.norm(x, 2, 1, keepdim=True), 
                            unc_hint_map[augmentation_basis_is_std][augmentation_basis][augmentation_auxiliary_is_std][augmentation_auxiliary]["l2"] )
                        
                        mAPs["aug"][augmentation_basis][augmentation_basis_is_std]["fv_weighting"][maximization_goal]["lambda"][augmentation_auxiliary][augmentation_auxiliary_is_std]["l2"] = lambda_weighting_mAPs
                        rank1s["aug"][augmentation_basis][augmentation_basis_is_std]["fv_weighting"][maximization_goal]["lambda"][augmentation_auxiliary][augmentation_auxiliary_is_std]["l2"] = lambda_weighting_rank1s



                        print(f"      computing performance for using a derived c weighted by a constant lambda in the feature vector weighting scheme... (entropy)")
                        
                        lambda_weighting_mAPs, lambda_weighting_rank1s = get_lambda_weighting_performances(
                            all_data, 
                            augmentation_basis, 
                            augmentation_basis_is_std, 
                            maximization_goal,
                            augmentation_auxiliary, 
                            augmentation_auxiliary_is_std, 
                            mAPs["vanilla"]["all"], 
                            rank1s["vanilla"]["all"],
                            lambda x: 2905.9861160031696 + 0.5 * torch.sum(torch.log(x), dim=1, keepdim=True),
                            unc_hint_map[augmentation_basis_is_std][augmentation_basis][augmentation_auxiliary_is_std][augmentation_auxiliary]["entropy"] )
                        
                        mAPs["aug"][augmentation_basis][augmentation_basis_is_std]["fv_weighting"][maximization_goal]["lambda"][augmentation_auxiliary][augmentation_auxiliary_is_std]["entropy"] = lambda_weighting_mAPs
                        rank1s["aug"][augmentation_basis][augmentation_basis_is_std]["fv_weighting"][maximization_goal]["lambda"][augmentation_auxiliary][augmentation_auxiliary_is_std]["entropy"] = lambda_weighting_rank1s"""

                        print(f"      computing performance for using a derived c weighted by a constant lambda in the feature vector weighting scheme... (1/l1(ln))")
                        
                        lambda_weighting_mAPs, lambda_weighting_rank1s = get_lambda_weighting_performances(
                            all_data, 
                            augmentation_basis, 
                            augmentation_basis_is_std, 
                            maximization_goal,
                            augmentation_auxiliary, 
                            augmentation_auxiliary_is_std, 
                            mAPs["vanilla"]["all"], 
                            rank1s["vanilla"]["all"],
                            lambda x: 1 / torch.norm(torch.log(x), 1, 1, keepdim=True),
                            unc_hint_map[augmentation_basis_is_std][augmentation_basis][augmentation_auxiliary_is_std][augmentation_auxiliary]["l1ln"] )
                        
                        mAPs["aug"][augmentation_basis][augmentation_basis_is_std]["fv_weighting"][maximization_goal]["lambda"][augmentation_auxiliary][augmentation_auxiliary_is_std]["1_l1ln"] = lambda_weighting_mAPs
                        rank1s["aug"][augmentation_basis][augmentation_basis_is_std]["fv_weighting"][maximization_goal]["lambda"][augmentation_auxiliary][augmentation_auxiliary_is_std]["1_l1ln"] = lambda_weighting_rank1s

                        print(f"      computing performance for using a derived c weighted by a constant lambda in the feature vector weighting scheme... (1/l1)")
                        
                        lambda_weighting_mAPs, lambda_weighting_rank1s = get_lambda_weighting_performances(
                            all_data, 
                            augmentation_basis, 
                            augmentation_basis_is_std, 
                            maximization_goal,
                            augmentation_auxiliary, 
                            augmentation_auxiliary_is_std, 
                            mAPs["vanilla"]["all"], 
                            rank1s["vanilla"]["all"],
                            lambda x: 1 / torch.norm(x, 1, 1, keepdim=True),
                            unc_hint_map[augmentation_basis_is_std][augmentation_basis][augmentation_auxiliary_is_std][augmentation_auxiliary]["l1"] )
                        
                        mAPs["aug"][augmentation_basis][augmentation_basis_is_std]["fv_weighting"][maximization_goal]["lambda"][augmentation_auxiliary][augmentation_auxiliary_is_std]["1_l1"] = lambda_weighting_mAPs
                        rank1s["aug"][augmentation_basis][augmentation_basis_is_std]["fv_weighting"][maximization_goal]["lambda"][augmentation_auxiliary][augmentation_auxiliary_is_std]["1_l1"] = lambda_weighting_rank1s

    output = {
        "mAPs": mAPs,
        "rank1s": rank1s
    }

    with open(out_path, "w") as f:
        json.dump(output, f)

    # execute~
    # get other results (non-UAL, BA-stuff)
    # TODO: send off



"""import subprocess
# code for getting the memory dump of the running job

def run_gdb(pid, output_file, gdb_path):
    # Construct the gdb commands
    gdb_commands = [
        f"attach {pid}",
        f"generate-core-file {output_file}",
        "detach",
        "quit"
    ]
    
    # Prepare the gdb command script
    gdb_script = "\n".join(gdb_commands)
    
    # Run gdb with the prepared script
    process = subprocess.Popen([gdb_path, '-q'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = process.communicate(input=gdb_script)
    
    # Print the output and error (if any)
    print("gdb stdout:", stdout)
    print("gdb stderr:", stderr)

# Define PID, output file, and gdb path
pid = 6155
output_file = "/data2/hpcuser/angel8547/dump2.gdb"
gdb_path = "/data2/hpcuser/angel8547/gdb/bin/gdb"

# Run the gdb memory dump
run_gdb(pid, output_file, gdb_path)
"""


        
def test_import():
    with open("all_UAL_results.json", "r") as f:
        def bool_keys_hook(d):
            new_d = {}
            for k, v in d.items():
                if k == 'true':
                    new_d[True] = v
                elif k == 'false':
                    new_d[False] = v
                else:
                    new_d[k] = v
            return new_d
        asd = json.load(f, object_hook=bool_keys_hook)
        print(asd)

#test_import()



def read_all_results(path="all_UAL_results.json"):

    with open(path, "r") as f:
        def bool_keys_hook(d):
            new_d = {}
            for k, v in d.items():
                if k == 'true':
                    new_d[True] = v
                elif k == 'false':
                    new_d[False] = v
                else:
                    new_d[k] = v
            return new_d
        all_ual_results = json.load(f, object_hook=bool_keys_hook)
    
    return all_ual_results

def tabulate_all_results_summary(path="all_UAL_results.json", precision=4, show_mAP=True, exclude_keys=[], exclude_paths_with_substr=[], path_sep="/"):
    """prints table of results under `path` with float precision `precision`. Shows mAP by default, set `show_mAP` to `False` for Rank-1.
    
    Filter by excluding all keys given under `exclude_keys` or by specifying parts of the path under `exclude_paths_with_substr`.
    
    Set `path_sep="']['"` to get nearly perfect copy-able dict-index-terms. NOTE: Bool-keys are actual bools as implemented.
    You can change this by removing the object_hook in `read_all_results` above.
    """



    results = read_all_results(path=path)

    def collect_paths_and_values(dic, path="") -> list[list]:

        rv = []
        max_max_val = float("-inf")
        max_avg_val = float("-inf")
        max_max_delta = float("-inf")
        max_avg_delta = float("-inf")
        max_max_val_path = ""
        max_avg_val_path = ""
        max_max_delta_path = ""
        max_avg_delta_path = ""

        for k,v in dic.items():

            if k in exclude_keys:
                continue

            new_path = path + path_sep + str(k)

            should_skip = False
            for ex_str in exclude_paths_with_substr:
                if ex_str in new_path:
                    should_skip = True
            if should_skip:
                continue

            if "all" in v.keys():
                # found a leaf
                rv.append([
                    f'{v["max"]:.{precision}f} ({v["mean"]:.{precision}f} ± {v["std"]:.{precision}f})',
                    f'{v["deltas"]["max"]:.{precision}f} ({v["deltas"]["mean"]:.{precision}f} ± {v["deltas"]["std"]:.{precision}f})',
                    new_path
                ])

                if float(v["max"]) > max_max_val:
                    max_max_val = float(v["max"])
                    max_max_val_path = new_path
                
                if float(v["mean"]) > max_avg_val:
                    max_avg_val = float(v["mean"])
                    max_avg_val_path = new_path

                if float(v["deltas"]["max"]) > max_max_delta:
                    max_max_delta = float(v["deltas"]["max"])
                    max_max_delta_path = new_path

                if float(v["deltas"]["mean"]) > max_avg_delta:
                    max_avg_delta = float(v["deltas"]["mean"])
                    max_avg_delta_path = new_path

            else:
                # recurse
                sub_rv, sub_best_stats = collect_paths_and_values(v, new_path)
                rv = rv + sub_rv

                if sub_best_stats["max_max_val"] > max_max_val:
                    max_max_val = sub_best_stats["max_max_val"]
                    max_max_val_path = sub_best_stats["max_max_val_path"]

                if sub_best_stats["max_avg_val"] > max_avg_val:
                    max_avg_val = sub_best_stats["max_avg_val"]
                    max_avg_val_path = sub_best_stats["max_avg_val_path"]

                if sub_best_stats["max_max_delta"] > max_max_delta:
                    max_max_delta = sub_best_stats["max_max_delta"]
                    max_max_delta_path = sub_best_stats["max_max_delta_path"]

                if sub_best_stats["max_avg_delta"] > max_avg_delta:
                    max_avg_delta = sub_best_stats["max_avg_delta"]
                    max_avg_delta_path = sub_best_stats["max_avg_delta_path"]


        best_stats = {
            "max_max_val": max_max_val,
            "max_max_val_path": max_max_val_path,
            "max_avg_val": max_avg_val,
            "max_avg_val_path": max_avg_val_path,
            "max_max_delta": max_max_delta,
            "max_max_delta_path": max_max_delta_path,
            "max_avg_delta": max_avg_delta,
            "max_avg_delta_path": max_avg_delta_path
        }

        return rv, best_stats
    
    mAP_summaries, best_mAPs = collect_paths_and_values(results["mAPs"])
    rank1_summaries, best_rank1s = collect_paths_and_values(results["rank1s"]) 

    from tabulate import tabulate

    # Define the headers separately
    headers = ["mAP: max (mean ± std)", "increase in mAP: same", "path"]

    # Tabulate the data with separate headers
    if show_mAP:
        table = tabulate(mAP_summaries, headers=headers, tablefmt='simple_grid')
        best_stats = best_mAPs
    else:
        table = tabulate(rank1_summaries, headers=headers, tablefmt='simple_grid')
        best_stats = best_rank1s

    print(table)
    print("") 
    print(f"Best Max mAP: {best_stats['max_max_val']:.{precision}f} (Path: {best_stats['max_max_val_path']})")
    print(f"Best Avg mAP: {best_stats['max_avg_val']:.{precision}f} (Path: {best_stats['max_avg_val_path']})")
    print(f"Best Max Delta: {best_stats['max_max_delta']:.{precision}f} (Path: {best_stats['max_max_delta_path']})")
    print(f"Best Avg Delta: {best_stats['max_avg_delta']:.{precision}f} (Path: {best_stats['max_avg_delta_path']})")


tabulate_all_results_summary(path="all_BoT_results.json", precision=4, exclude_keys=["best_avg_res-mAP"])
#generate_uncertainty_score_distribution_plots()
# get_all_performances(
#     [f"/data2/hpcuser/angel8547/results_backup_2024_03_13/UAL/{i}/raw_model_outputs.json" for i in list(range(11,20)) + list(range(21, 30)) + [55,57]],
#     #[f"/data2/hpcuser/angel8547/DNet/{i}/raw_model_outputs.json" for i in range(20)],
#     #[f"/data2/hpcuser/angel8547/results_backup_2024_03_13/UAL/11/raw_model_outputs.json"],
#     ["dist"],#, "data", "dist"],
#     "UAL_lambda_B_dist.json"
# )  # bsub -w "done(2642293)" -gpu "num=1" -J "UAL_dist" -q "BatchGPU" -outdir . -o ./out.%J -e ./err.%J python ../../misc/data_stats.py 
  # bsub -J "dist_B72" -q "Batch72" -outdir . -o ./lout.%J -e ./lerr.%J python ../../misc/data_stats2.py 
# -R "select[hname!='makalu94']"
# -w "done(2643946)"
