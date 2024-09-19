import json
import numpy as np

def load_data(path="/data2/hpcuser/angel8547/results_backup_2024_03_13/UAL/28/raw_model_outputs.json"):#DATA_FILENAME):
    print(f"loading {path}...")
    with open(path, 'r') as data_file:
        data = json.load(data_file)
    print("DONE!")

    return data

def get_vectors(data, set_id, vector_type):
    """Returns the requested raw model output vectors for the set in a list."""
    return np.array([data['data'][name][vector_type]
            for name in sorted(data['sets'][set_id])])

bad_nums = []
total_nums = []
all_alphas = []
max_alphas = []
for path in [f"/data2/hpcuser/angel8547/results_backup_2024_03_13/UAL/{i}/raw_model_outputs.json" for i in list(range(11,20)) + list(range(21, 30)) + [55,57]]:
    data = load_data(path)

    mod_unc_vecs = np.concatenate((get_vectors(data, "G", "variance_of_mean_vector"), get_vectors(data, "Q", "variance_of_mean_vector")), axis=0)

    print(mod_unc_vecs.shape)

    num_bad_vecs = 0
    alphas = []
    for vec in mod_unc_vecs:
        # print(vec)
        # print(vec > 1)
        # print(sum(vec>1))
        if sum(vec > 1) > 0:
            num_bad_vecs += 1

        alpha = np.linalg.norm(np.log(vec), 1, 0) / np.abs(np.sum(np.log(vec), 0))
        alphas.append(alpha)
    
    print(f"Got {num_bad_vecs=} out of {len(mod_unc_vecs)} which is {100*num_bad_vecs/len(mod_unc_vecs):.2f}% over Q+G in {path}")
    print(f"{np.max(alphas)=:.4f}")
    all_alphas.append(alphas)
    max_alphas.append(np.max(alphas))

    bad_nums.append(num_bad_vecs)
    total_nums.append(len(mod_unc_vecs))

print(f"Total average: {100* sum(bad_nums) / sum(total_nums):.2f}%")
print(f"Worst percentage: {100*np.max([b/t for b,t in zip(bad_nums, total_nums)]):.2f}%")
#print(f"Percent-Average: {100*np.mean([b/t for b,t in zip(bad_nums, total_nums)]):.2f}%")
print(f"Absolute worst alpha: {np.max(max_alphas)}")
avg_alphas = []
for alphas in all_alphas:
    avg_alpha = np.mean([alpha for alpha in alphas if alpha != 1.0])
    avg_alphas.append(avg_alpha)
    print(f"Average Alpha in cases where it is not 1: {avg_alpha}")
print(f"Over all average alpha in cases where it is not 1: {np.mean([alpha for alpha in np.array(all_alphas).flatten() if alpha != 1.0])}")