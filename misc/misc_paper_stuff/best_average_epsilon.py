"""
The purpose of this file is to find that epsilon which on average across the different runs results in the best mAP.
We do this by parsing the values from the log file.
"""

import numpy as np
import matplotlib.pyplot as plt
import json

from sklearn.neighbors import KernelDensity


LOG_FILE_PATH = 'all_hyper.log'



def get_vectors(data, set_id, vector_type):
    """Returns the requested raw model output vectors for the set in a list."""
    return [data['data'][name][vector_type]
            for name in sorted(data['sets'][set_id])]



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


def main() -> None:
    # read log file
    with open(LOG_FILE_PATH, "r") as f:
        log = f.read()

    log_sections = log.split("loading /usr/scratch4/angel8547/results/UAL/")[1:] # discard first section as it contains nothing
    
    all_data_points = {}
    for sec in log_sections:
        data_point_strings = sec.split("\nmAP(epsilon=")[1:]
        data_point_strings[-1] = data_point_strings[-1].split("\n")[0] # remove rest of string we don't need
        
        data_points = [s.split(") = ") for s in data_point_strings] # data_points = [[eps, mAP], ...]
        for eps, mAP in data_points:
            if eps not in all_data_points.keys():
                all_data_points[eps] = []
            all_data_points[eps].append(float(mAP))
    # all_data_points: {eps: [mAP_run_1, mAP_run_2, ...], ...}
            

    def best_lambdas(map):
        reverse_map = [[] for _ in map["0"]]
        lambda_list = []
        for k, v in map.items():
            lambda_list.append(k)
            for i, mAP in enumerate(v):
                reverse_map[i].append(mAP)
        best_lambdas = [float(lambda_list[i]) for i in [mAPs_for_lambdas_for_a_run.index(max(mAPs_for_lambdas_for_a_run)) for mAPs_for_lambdas_for_a_run in reverse_map]]
        print(np.max(reverse_map))
        return best_lambdas
    
    def get_stats_container(float_list):
        """upgrade your list to a dict containing the list as well as some statistics about it"""
        return {
            "min": np.min(float_list),
            "mean": np.mean(float_list),
            "std": np.std(float_list),
            "max": np.max(float_list),
            "all": float_list
        }

    print(get_stats_container(best_lambdas(all_data_points)))
    exit()


    # 27,28,29
    #print(all_data_points["0"][15:18])

    #run_index = 16
    for run_index, path in zip(list(range(18)), [f"/data2/hpcuser/angel8547/results_backup_2024_03_13/UAL/{i}/raw_model_outputs.json" for i in [11,12,13,14,15,16,17,18,19,21,22,23,24,25,26,27,28,29]]):

        plot_epss = []
        plot_mAPs = []
        #print(path)

        for eps_s, mAPs in all_data_points.items():
            eps = float(eps_s)
            plot_epss.append(eps)
            mAP = mAPs[run_index]
            plot_mAPs.append(mAP)
        #plt.scatter(plot_epss, plot_mAPs, marker="x", color="k")
        #plt.plot(plot_epss, plot_mAPs)


        # load data
        data = load_data(path)
        # get model uncertainty vectors
        unc_vecs = get_vectors(data, "Q", "variance_of_mean_vector")
        # take sqrt
        unc_vecs = np.sqrt(unc_vecs)
        # compute function
        eps_comp = 1024 / np.sum(np.abs(np.log(unc_vecs)), axis=1)
        # perform KDE
        x_values = eps_comp
        x_mean = np.mean(x_values)

        print(f"{x_mean=}")
        print(f"best epsilon this run = {plot_epss[plot_mAPs.index(max(plot_mAPs))]}")
        print(f"ratio: {plot_epss[plot_mAPs.index(max(plot_mAPs))]/x_mean}, {1024*plot_epss[plot_mAPs.index(max(plot_mAPs))]/x_mean}")
        x_std = np.std(x_values)
        x_min = x_max=None
        if x_min is None or x_max is None:
            x_min = np.min(x_values)
            x_max = np.max(x_values)
            diff_x = x_max - x_min
        else:
            diff_x = x_max - x_min
        
        kde = KernelDensity(bandwidth=(diff_x*0.03), kernel='gaussian')
        kde.fit(np.array(x_values)[:, np.newaxis])
        x = np.linspace(x_mean - 7 * x_std,
                        x_mean + 7 * x_std, 100)
        y = np.exp(kde.score_samples(x[:, np.newaxis]))
        # plot KDE with second y axis



        fig, ax1 = plt.subplots()



        # Plot the first dataset on the first y-axis
        ax1.scatter(plot_epss, plot_mAPs, marker='x', color='k', label='y1', linewidths=0.2)
        ax1.plot(plot_epss, plot_mAPs, color='blue')
        ax1.set_xlabel('eps')
        ax1.set_ylabel('mAP', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')

        # Create a second y-axis sharing the same x-axis
        ax2 = ax1.twinx()
        #ax2.scatter(x_values, y2_values, marker='o', color='red', label='y2')
        ax2.plot(x, y, color='red')
        ax2.set_ylabel('probability density', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        ax2.set_ylim(bottom=0)

        ax2.plot([x_mean, x_mean], [0, 3.5], color="red")

        # Add title
        plt.title('mAP vs epsilon + probability of epsilon computed without magic')

        # Show plot
        fig.tight_layout()
        plt.savefig(f"eps_vs_map+prob_{run_index}.pdf")
        plt.close(fig)

    exit()



            
    best_eps = "0"
    best_avg_mAP = 0 
    best_avg_delta = 0 # increases in mAP by utilizing our method
    base_mAPs = all_data_points["0"] # mAPs of the UAL baseline runs
    for eps, mAP_list in all_data_points.items():
        #avg_mAP = np.mean(mAP_list)
        deltas = [ b - a for a, b in zip(base_mAPs, mAP_list)]
        avg_delta = np.mean(deltas)
        print(f"For epsilon={eps}, the average delta is {avg_delta}")
        if avg_delta > best_avg_delta:
            best_avg_delta = avg_delta
            best_eps = eps 

    print("\n\n") # should we care about the increase in mAP (as implemented) or final mAP? does it make a difference?
    print(f"The epsilon that results in the greatest incerase in mAP on average over the different runs is {best_eps} with an average delta of {best_avg_delta}.")
    deltas = [ b - a for a, b in zip(base_mAPs, all_data_points[best_eps])]
    best_mAPs = all_data_points["0.23521471862576143"] # mAPs for the average best epsilon
    print(f"{max(best_mAPs)=}") # if we want an even bigger maximum, we can also report the max of the fine-tuned hyperparameters but I wouldn't do that tbh
    #print(f"{best_mAPs=}")
    #print(f"Using this epsilon, the minimum observed increase in mAP is {min(deltas)} and the maximum increase is {max(deltas)}")
    print(f"eps={best_eps}: delta: min={min(deltas)}, max={max(deltas)}, mean={np.mean(deltas)}, std={np.std(deltas)}")

    
    
    

if __name__ == "__main__":
    main()