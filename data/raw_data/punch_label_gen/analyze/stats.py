import os
import numpy as np
import pandas as pd

from common.utils import retrieve_name


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


def gen_punch_stats(data_directory):
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

    punch_count_print = [count_r_fwd, count_r_rev, count_l_fwd, count_l_rev]
    names = list(map(retrieve_name, punch_count_print))
    punch_stats = []

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
        print(dir_stats)

    return punch_stats


if __name__ == '__main__':
    INPUT_BASE_PATH = os.path.join("data", "raw_data")
    data_dir = os.path.join(INPUT_BASE_PATH, "punch_label_gen", "punch_label", "tertiary")
    # data_dir = os.path.join(INPUT_BASE_PATH, "punch_label_gen", "punch_label", "binary")
    gen_punch_stats(data_dir)
