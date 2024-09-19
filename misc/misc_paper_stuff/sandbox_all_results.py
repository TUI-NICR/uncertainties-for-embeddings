import json
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

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


# for k,v in data["sets"].items():
#     print(f"{k} {len(v)}")
# exit()

def get_vectors(data, set_id, vector_type):
    """Returns the requested raw model output vectors for the set in a list."""
    return [data['data'][name][vector_type]
            for name in sorted(data['sets'][set_id])]

def get_QG_vecs(data, vector_type):
    """wrapper for getting query and gallery vectors to avoid boilerplate"""
    return torch.Tensor(get_vectors(data, "Q", vector_type)), torch.Tensor(get_vectors(data, "G", vector_type))
    #return torch.Tensor(get_vectors(data, "Q", vector_type)).cuda(), torch.Tensor(get_vectors(data, "G", vector_type)).cuda()


all_data = []
for path in [f"/data2/hpcuser/angel8547/results_backup_2024_03_13/UAL/{i}/raw_model_outputs.json" for i in list(range(11,20)) + list(range(21, 30)) + [55, 57]]:
    data = load_data(path)
    all_data.append(data)
    break # TODO remove



unc_hint_map = { # basis_is_std, basis, aux_is_std, aux
    True: {
        "model": {
            True: {
                "model": {
                    "entropy": 0,
                    "l1ln": 0,
                    "l1": 0,
                    "l2": 0
                },
                "data": {
                    "entropy": 0,
                    "l1ln": 0,
                    "l1": 0,
                    "l2": 0
                },
                "dist": {
                    "entropy": 0,
                    "l1ln": 0,
                    "l1": 0,
                    "l2": 0
                }
            },
            False: {
                "model": {
                    "entropy": 0,
                    "l1ln": 0,
                    "l1": 0,
                    "l2": 0
                },
                "data": {
                    "entropy": 0,
                    "l1ln": 0,
                    "l1": 0,
                    "l2": 0
                },
                "dist": {
                    "entropy": 0,
                    "l1ln": 0,
                    "l1": 0,
                    "l2": 0
                }
            }
        },
        "data": {
            True: {
                "model": {
                    "entropy": 0,
                    "l1ln": 0,
                    "l1": 0,
                    "l2": 0
                },
                "data": {
                    "entropy": 0,
                    "l1ln": 0,
                    "l1": 0,
                    "l2": 0
                },
                "dist": {
                    "entropy": 0,
                    "l1ln": 0,
                    "l1": 0,
                    "l2": 0
                }
            },
            False: {
                "model": {
                    "entropy": 0,
                    "l1ln": 0,
                    "l1": 0,
                    "l2": 0
                },
                "data": {
                    "entropy": 0,
                    "l1ln": 0,
                    "l1": 0,
                    "l2": 0
                },
                "dist": {
                    "entropy": 0,
                    "l1ln": 0,
                    "l1": 0,
                    "l2": 0
                }
            }
        },
        "dist": {
            True: {
                "model": {
                    "entropy": 0,
                    "l1ln": 0,
                    "l1": 0,
                    "l2": 0
                },
                "data": {
                    "entropy": 0,
                    "l1ln": 0,
                    "l1": 0,
                    "l2": 0
                },
                "dist": {
                    "entropy": 0,
                    "l1ln": 0,
                    "l1": 0,
                    "l2": 0
                }
            },
            False: {
                "model": {
                    "entropy": 0,
                    "l1ln": 0,
                    "l1": 0,
                    "l2": 0
                },
                "data": {
                    "entropy": 0,
                    "l1ln": 0,
                    "l1": 0,
                    "l2": 0
                },
                "dist": {
                    "entropy": 0,
                    "l1ln": 0,
                    "l1": 0,
                    "l2": 0
                }
            }
        }
    },
    False: {
        "model": {
            True: {
                "model": {
                    "entropy": 0,
                    "l1ln": 0,
                    "l1": 0,
                    "l2": 0
                },
                "data": {
                    "entropy": 0,
                    "l1ln": 0,
                    "l1": 0,
                    "l2": 0
                },
                "dist": {
                    "entropy": 0,
                    "l1ln": 0,
                    "l1": 0,
                    "l2": 0
                }
            },
            False: {
                "model": {
                    "entropy": 0,
                    "l1ln": 0,
                    "l1": 0,
                    "l2": 0
                },
                "data": {
                    "entropy": 0,
                    "l1ln": 0,
                    "l1": 0,
                    "l2": 0
                },
                "dist": {
                    "entropy": 0,
                    "l1ln": 0,
                    "l1": 0,
                    "l2": 0
                }
            }
        },
        "data": {
            True: {
                "model": {
                    "entropy": 0,
                    "l1ln": 0,
                    "l1": 0,
                    "l2": 0
                },
                "data": {
                    "entropy": 0,
                    "l1ln": 0,
                    "l1": 0,
                    "l2": 0
                },
                "dist": {
                    "entropy": 0,
                    "l1ln": 0,
                    "l1": 0,
                    "l2": 0
                }
            },
            False: {
                "model": {
                    "entropy": 0,
                    "l1ln": 0,
                    "l1": 0,
                    "l2": 0
                },
                "data": {
                    "entropy": 0,
                    "l1ln": 0,
                    "l1": 0,
                    "l2": 0
                },
                "dist": {
                    "entropy": 0,
                    "l1ln": 0,
                    "l1": 0,
                    "l2": 0
                }
            }
        },
        "dist": {
            True: {
                "model": {
                    "entropy": 0,
                    "l1ln": 0,
                    "l1": 0,
                    "l2": 0
                },
                "data": {
                    "entropy": 0,
                    "l1ln": 0,
                    "l1": 0,
                    "l2": 0
                },
                "dist": {
                    "entropy": 0,
                    "l1ln": 0,
                    "l1": 0,
                    "l2": 0
                }
            },
            False: {
                "model": {
                    "entropy": 0,
                    "l1ln": 0,
                    "l1": 0,
                    "l2": 0
                },
                "data": {
                    "entropy": 0,
                    "l1ln": 0,
                    "l1": 0,
                    "l2": 0
                },
                "dist": {
                    "entropy": 0,
                    "l1ln": 0,
                    "l1": 0,
                    "l2": 0
                }
            }
        }
    }
}


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
all_results_json = {}
with open("UAL_results_model.json", "r") as f:
    all_results_json["variance_of_mean_vector"] = json.load(f, object_hook=bool_keys_hook)

with open("UAL_results_data.json", "r") as f:
    all_results_json["variance_vector"] = json.load(f, object_hook=bool_keys_hook)

with open("UAL_results_dist.json", "r") as f:
    all_results_json["variance_of_variance_vector"] = json.load(f, object_hook=bool_keys_hook)

for do_sqrt in [True, False]:
    print(f"********************************* sqrt: {do_sqrt}")


    mAPs = []
    mAP_deltas = []
    rank1s = []
    rank1_deltas = []

    for data in [all_data[0]]:# TODO: change back to all
        #modunc = get_vectors(data, "Q", "variance_of_mean_vector")



        #modunc = np.sqrt(modunc)

        # ls = 1024 / np.sum(np.abs(np.log(modunc)), axis=1)
        # print(ls[10:20])
        # print("mean ls", np.mean(ls))


        # lsa = 1024 / np.sum(np.abs(np.log(np.mean(modunc, axis=0))))

        # print(f"{lsa=}")



        # want to compare: vanilla, eps=lsa und eps=f(mod)



        for uncertainty_type in tqdm(["variance_of_mean_vector", "variance_of_variance_vector", "variance_vector"]):
            # query_unc_vecs, gallery_unc_vecs = get_QG_vecs(data, uncertainty_type)
            # query_fvs, gallery_fvs = get_QG_vecs(data, "mean_vector")
                    
            # if do_sqrt:
            #     query_unc_vecs = torch.sqrt(query_unc_vecs)
            #     gallery_unc_vecs = torch.sqrt(gallery_unc_vecs)

            # vdistmat = compute_cosine_distance(query_fvs, gallery_fvs)
            # vanilla = get_mAP(vdistmat, data)
            # vanilla_rank1 = get_rank1(vdistmat, data)




            entropy = lambda x: torch.mean(2905.9861160031696 + 0.5 * torch.sum(torch.log(x), dim=1, keepdim=True))
            l1ln = lambda x: torch.mean(1 / torch.norm(torch.log(x), 1, 1, keepdim=True))
            l1 = lambda x: torch.mean(1 / torch.norm(x, 1, 1, keepdim=True))
            l2 = lambda x: torch.mean(torch.norm(x, 2, 1, keepdim=True))

            # c_opt = (0.12 * (1+do_sqrt))
            name_map = {
                "variance_of_mean_vector": "model", 
                "variance_of_variance_vector": "dist", 
                "variance_vector": "data"
            }
            c_opt = all_results_json[uncertainty_type]["mAPs"]["aug"][name_map[uncertainty_type]][do_sqrt]["fv_weighting"]["best_avg_delta-mAP"]["eps"]["eps"]

            for do_sqrt_aux in [True, False]:
                for aux_uncertainty_type in ["variance_of_mean_vector", "variance_of_variance_vector", "variance_vector"]:
                    query_unc_vecs, gallery_unc_vecs = get_QG_vecs(data, aux_uncertainty_type)
                    query_fvs, gallery_fvs = get_QG_vecs(data, "mean_vector")
                            
                    if do_sqrt_aux:
                        query_unc_vecs = torch.sqrt(query_unc_vecs)
                        gallery_unc_vecs = torch.sqrt(gallery_unc_vecs)

                    # print(f"{uncertainty_type=}, {do_sqrt=}")
                    # print(f"Q: l(e): {c_opt/entropy(query_unc_vecs)}, l(l1ln): {c_opt/l1ln(query_unc_vecs)}, l(l1): {c_opt/l1(query_unc_vecs)}, l(l2): {c_opt/l2(query_unc_vecs)}")
                    #print(f"G: l(e): {c_opt/entropy(gallery_unc_vecs)}, l(l1ln): {c_opt/l1ln(gallery_unc_vecs)}, l(l1): {c_opt/l1(gallery_unc_vecs)}")
                    unc_hint_map[do_sqrt][name_map[uncertainty_type]][do_sqrt_aux][name_map[aux_uncertainty_type]]["entropy"] = (c_opt/entropy(query_unc_vecs)).item()
                    unc_hint_map[do_sqrt][name_map[uncertainty_type]][do_sqrt_aux][name_map[aux_uncertainty_type]]["l1ln"] = (c_opt/l1ln(query_unc_vecs)).item()
                    unc_hint_map[do_sqrt][name_map[uncertainty_type]][do_sqrt_aux][name_map[aux_uncertainty_type]]["l1"] = (c_opt/l1(query_unc_vecs)).item()
                    unc_hint_map[do_sqrt][name_map[uncertainty_type]][do_sqrt_aux][name_map[aux_uncertainty_type]]["l2"] = (c_opt/l2(query_unc_vecs)).item()


print("\n\n")
with open("hint_values.json", "w") as f:
    print(json.dump(unc_hint_map, f, indent=4))

exit()
def asdd():
    def qweqwe():
        """print("variant 1")

        query_fv_1 = torch.Tensor(query_fvs / (query_unc_vecs + lsa))
        gallery_fv_1 = torch.Tensor(gallery_fvs / (gallery_unc_vecs + lsa))


        var1 = get_mAP(compute_cosine_distance(query_fv_1, gallery_fv_1), data)"""

        print("variant 2") # TODO: change back to 1024
        epsQ = 1024 / torch.sum(torch.abs(torch.log(query_unc_vecs)), axis=1, keepdim=True) 
        epsG = 1024 / torch.sum(torch.abs(torch.log(gallery_unc_vecs)), axis=1, keepdim=True) 

        query_fv_2 = torch.Tensor(query_fvs / (query_unc_vecs + epsQ))
        gallery_fv_2 = torch.Tensor(gallery_fvs / (gallery_unc_vecs + epsG))

        distmat = compute_cosine_distance(query_fv_2, gallery_fv_2)

        var2 = get_mAP(distmat, data)
        mAPs.append(var2)
        # mAP_deltas.append(var2 - vanilla)

        var2_rank1 = get_rank1(distmat, data)
        rank1s.append(var2_rank1)
        # rank1_deltas.append(var2_rank1 - vanilla_rank1)

        # print(f"{vanilla=}, {var1=}, {var2=}")

    print("-------------------------------mAP")
    print(json.dumps(get_stats_and_delta_container(mAPs, mAP_deltas)))
    print("-------------------------------rank1:")
    print(json.dumps(get_stats_and_delta_container(rank1s, rank1_deltas)))


exit()
import struct
import pickle
from tqdm import tqdm

# Define the known float values from the "mean_vector"
known_floats = [
    -0.043920565, 0.4758922, -0.09950168, -0.039915495, 0.250147,
    0.33761954, 0.16501793, 0.5450358, 0.18995231, -0.15210721,
    0.0919457, 0.24017018, -0.18332021, -0.033207517, 0.058932047,
    0.25821987, 0.049829923, -0.13177775, 0.5012016, 0.3927497,
    -0.23414876, 0.23948748, 0.17459801, 0.10393681, 0.031443313,
    -0.12538798, 0.14633623, 0.10605848, 0.21705183, 0.027758693,
    0.13551636, -0.061050102, -0.22891988, -0.26548633, 0.03204878,
    -0.035698947, 0.042041678, 0.15962255, 0.053588055, 0.19235678,
    0.018856173, -0.08228459, 0.013467914, -0.069258586
]

float_format = struct.Struct('f')  # Format for a single float (4 bytes)

def search_floats_in_memory(memory, known_floats):
    positions = []
    float_bytes = [float_format.pack(f) for f in known_floats]

    for i in tqdm(range(10883068703, len(memory) - len(float_bytes) * 4 + 1)):
        if all(memory[i + j * 4:i + (j + 1) * 4] == float_bytes[j] for j in range(len(float_bytes))):
            positions.append(i)
            break
    
    return positions

def extract_dict_from_memory(memory, start_position):
    # Define how you will extract the data structure based on its layout
    # This example assumes a simple structure and requires adjustment
    extracted_data = {}
    offset = start_position

    # Example extraction of a dictionary with known structure
    mean_vector = []
    for _ in tqdm(range(len(known_floats))):
        float_bytes = memory[offset:offset+4]
        value = float_format.unpack(float_bytes)[0]
        mean_vector.append(value)
        offset += 4

    # Adjust the following code based on your actual data structure
    extracted_data["0300_c5s1_067148_00.jpg"] = {"mean_vector": mean_vector}
    return extracted_data

# Read the memory dump
with open('/data2/hpcuser/angel8547/dump.gdb', 'rb') as f:
    memory = f.read()
print("loaded!")
# Search for known floats
positions = search_floats_in_memory(memory, known_floats)

print(f"{positions=}")
exit()


# Extract and save data if known floats are found
if positions:
    for pos in positions:
        extracted_data = extract_dict_from_memory(memory, pos)
        with open(f'/path/to/extracted_data_{pos}.pkl', 'wb') as f:
            pickle.dump(extracted_data, f)
    print(f"Data extracted at positions: {positions}")
else:
    print("Known float values not found in the memory dump.")














exit()
import wandb
api = wandb.Api()
import numpy as np
import json

def get_stats_container(float_list):
    """upgrade your list to a dict containing the list as well as some statistics about it"""
    return {
        "min": np.min(float_list),
        "mean": np.mean(float_list),
        "std": np.std(float_list),
        "max": np.max(float_list),
        "all": float_list
    }

# Project is specified by <entity/project-name>
runs = api.runs("ba_ue/mod_arch")

mAPs = []
rank1s = []

for run in runs: 
    if run.config["run_type"] == "dnet_bb-naive_naive":
        #print("got one")
        print(len(run.history()))
        history = run.history()
        if len(history) == 0:
            continue

        max_mAP = max([history[i]["eval.cosine.mAP"] for i in range(len(history)) if history[i]["eval.cosine.mAP"] != None])
        
        for i in range(len(history)):
            if history[i]["eval.cosine.mAP"] == max_mAP:
                max_mAP_i = i

        rank1 = history[max_mAP_i]["eval.cosine.Rank-1"]

        #print(max_mAP, "//", rank1)
        mAPs.append(max_mAP)
        rank1s.append(rank1)
        
        continue
    else:
        continue


print(len(mAPs), mAPs)
print(len(rank1s), rank1s)



plain = {
    "mAPs": get_stats_container(mAPs),
    "rank1s": get_stats_container(rank1s)
}

print("------------------------------------")

runs = api.runs("ba_ue/mod_arch-DNet_experiment3")

mAPs = []
rank1s = []

for run in runs: 
    if run.config["run_type"] == "dnet_RPCT":
        #print("got one")
        print(len(run.history()))
        history = run.history()
        if len(history) == 0:
            continue

        max_mAP = max([history[i]["eval.cosine.mAP"] for i in range(len(history)) if history[i]["eval.cosine.mAP"] != None])
        
        for i in range(len(history)):
            if history[i]["eval.cosine.mAP"] == max_mAP:
                max_mAP_i = i

        rank1 = history[max_mAP_i]["eval.cosine.Rank-1"]

        #print(max_mAP, "//", rank1)
        mAPs.append(max_mAP)
        rank1s.append(rank1)
        
        continue
    else:
        continue


print(len(mAPs), mAPs)
print(len(rank1s), rank1s)



rptc = {
    "mAPs": get_stats_container(mAPs),
    "rank1s": get_stats_container(rank1s)
}


out = {
    "plain": plain,
    "rptc": rptc
}

print(out)


with open("all_DNet_results.json", "w") as f:
    json.dump(out, f)











exit()

BoT_mAPs = [86.24864220619202,
                            86.42894625663757,
                            86.5966260433197,
                            86.3450288772583,
                            86.52341365814209,
                            86.40053868293762,
                            86.42350435256958,
                            86.35214567184448,
                            86.4232063293457,
                            86.40605807304382]

BoT_rank1s = [94.2992866039276,
                            94.44774389266968,
                            94.86342072486876,
                            94.0914511680603,
                            94.59620118141174,
                            94.0914511680603,
                            94.53681707382202,
                            94.38835978507996,
                            94.59620118141174,
                            94.269597530365]

import numpy as np
import json

def get_stats_container(float_list):
    """upgrade your list to a dict containing the list as well as some statistics about it"""
    return {
        "min": np.min(float_list),
        "mean": np.mean(float_list),
        "std": np.std(float_list),
        "max": np.max(float_list),
        "all": float_list
    }

BoT_results = {
    "mAPs": get_stats_container(BoT_mAPs),
    "rank1s": get_stats_container(BoT_rank1s)
}

with open("all_BoT_results.json", "w") as f:
    json.dump(BoT_results, f)