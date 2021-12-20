import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 14})
from common.utils import retrieve_name


def get_bins(min_max):
    d = min_max[1] - min_max[0]
    num_bins = 6
    step = d / num_bins
    bins = [round(min_max[0] + step * i, 2) for i in range(num_bins)]
    return np.array(bins)


def get_binned_arr(arr, arr_min_max):
    bins = get_bins(arr_min_max)
    bin_idx = np.digitize(arr, bins)

    binned_arr = []
    for idx in bin_idx:
        binned_arr.append(bins[idx - 1])
    binned_arr = np.array(binned_arr)
    return binned_arr, bin_idx


def get_punch_count(punch_labels_df, hand, punch_val):
    condition = np.array(punch_labels_df[hand] == punch_val)

    match_val = True
    count_array = np.diff(np.where(np.concatenate(([condition[0]],
                                                   condition[:-1] != condition[1:],
                                                   [match_val])))[0])[::2]

    # print('###############################################################################')
    # print('hand: ', hand, ',dir: ', punch_val)
    # print('len of count: ', len(count_array))
    # print('\n')
    # print('******************************************************************************')

    return count_array


def gen_punch_stats(data_directory, plot=False):
    count_r_fwd = np.array([])
    count_r_rev = np.array([])
    count_l_fwd = np.array([])
    count_l_rev = np.array([])
    for file in os.listdir(data_directory):
        filename = os.path.join(data_directory, file)
        punch_phase_df = pd.read_csv(filename)
        count_r_fwd = np.append(count_r_fwd, get_punch_count(punch_phase_df, 'right punch', 1.0))
        count_r_rev = np.append(count_r_rev, get_punch_count(punch_phase_df, 'right punch', -1.0))
        count_l_fwd = np.append(count_l_fwd, get_punch_count(punch_phase_df, 'left punch', 1.0))
        count_l_rev = np.append(count_l_rev, get_punch_count(punch_phase_df, 'left punch', -1.0))

    count_r = list(np.array(count_r_fwd) + np.array(count_r_rev))
    count_l = list(np.array(count_l_fwd) + np.array(count_l_rev))

    r_vals, r_bin_idxs = get_binned_arr(count_r, [0, 60])
    l_vals, l_bin_idxs = get_binned_arr(count_l, [0, 60])

    r_unique, r_counts = np.unique(count_r, return_counts=True)
    r_count_dict = {k: v for k, v in zip(r_unique, r_counts)}
    l_unique, l_counts = np.unique(l_vals, return_counts=True)
    l_count_dict = {k: v for k, v in zip(l_unique, l_counts)}

    if plot:

        plt.hist(count_r, density=False, bins=60, color ="gold",edgecolor='black', linewidth=1.2)  # density=False would make counts
        plt.ylabel('Num Punches')
        plt.xlabel('Frames')
        plt.title("Count of right punches in dataset")
        plt.xlim(0, 60)
        plt.ylim(0, 40)
        plt.axvline(sum(count_r)/len(count_r), color='black', linestyle='--', label='Avg right punch ='+str(round(sum(count_r)/len(count_r)))+ " frames")
        plt.legend(loc="upper left")
        plt.savefig("data/raw_data/punch_label_gen/analyze/right_punch_count.png", dpi=300)
        plt.clf()

        plt.hist(count_l, density=False, bins=60, color="cyan",edgecolor='black', linewidth=1.2)  # density=False would make counts
        plt.ylabel('Num Punches')
        plt.xlabel('Frames')
        plt.title("Count of left punches in dataset")
        plt.xlim(0, 60)
        plt.ylim(0, 40)

        plt.axvline(sum(count_l) / len(count_l), color='black', linestyle='--',
                    label='Avg left punch =' + str(round(sum(count_l) / len(count_l))) + " frames")
        plt.legend(loc="upper left")
        plt.savefig("data/raw_data/punch_label_gen/analyze/left_punch_count.png", dpi=300)
        plt.clf()


    punch_count_print = [count_r_fwd, count_r_rev, count_l_fwd, count_l_rev, count_r, count_l]
    names = list(map(retrieve_name, punch_count_print))
    punch_stats = []

    avg_punch = []
    for i, p_count in enumerate(punch_count_print):
        p_count = np.array(p_count)

        dir_stats = {
            "name": names[i],
            "num_punches": len(p_count),
            "min_punch_duration": p_count.min(),
            "mean_punch_duration": p_count.mean(),
            "median_punch_duration": np.median(p_count),
            "max_punch_duration": p_count.max()
        }
        punch_stats.append(dir_stats)
        avg_punch.append(p_count.mean())
        print(dir_stats)

    # print("Overall Average Punch Duration = ", round(np.sum(avg_punch[:4])/2))
    print("Overall Average Punch Duration = ", round(np.sum(avg_punch[4:]) / 2))

    return punch_stats


if __name__ == '__main__':
    INPUT_BASE_PATH = os.path.join("data", "raw_data")
    data_dir = os.path.join(INPUT_BASE_PATH, "punch_label_gen", "punch_label", "tertiary")
    # data_dir = os.path.join(INPUT_BASE_PATH, "punch_label_gen", "punch_label", "binary")
    punch_stats = gen_punch_stats(data_dir, plot=True)
    print(punch_stats)
    p_df = pd.DataFrame(punch_stats)
    print(p_df.to_latex(index=False))
