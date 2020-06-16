import numpy as np
import pandas as pd
from scipy import stats

binary_punch_phase_df = pd.read_csv("Blender Code Snippets/data annotation res/test3.csv")


# condition = np.array([True, True, True, False, False, True, True, False, True, True, True, True, False, False, True])
# condition = np.array([1,1,0,1,1,1,1,0,1])


def get_punch_count(hand, punch_val):
    # punch_val = 1.0
    condition = np.array(binary_punch_phase_df[hand] == punch_val)
    # print(condition)
    # condition = np.array(binary_punch_phase_df['left punch'] == punch_val)
    match_val = True
    count_array = np.diff(np.where(np.concatenate(([condition[0]],
                                                   condition[:-1] != condition[1:],
                                                   [match_val])))[0])[::2]

    print('hand: ', hand, ',dir: ', punch_val)
    print('count: ', count_array)
    # print(count_array.sum() / len(count_array))
    print('mean: ', count_array.mean())
    print('median: ', np.median(count_array))
    # print('mode: ', stats.mode(count_array))

    unique_elements, counts_elements = np.unique(count_array, return_counts=True)
    print("Frequency of unique values:")
    el_count = {k: v for k, v in zip(unique_elements, counts_elements)}
    print(el_count)
    # print(np.asarray((unique_elements, counts_elements)))

    print('\n')


get_punch_count('right punch', 1.0)
get_punch_count('right punch', -1.0)
get_punch_count('left punch', 1.0)
get_punch_count('left punch', -1.0)

# punch_val = 1.0
# condition = np.array(binary_punch_phase_df['right punch'] == punch_val)
# print(condition)
# # condition = np.array(binary_punch_phase_df['left punch'] == punch_val)
# match_val = True
# count_array = np.diff(np.where(np.concatenate(([condition[0]],
#                                                condition[:-1] != condition[1:],
#                                                [match_val])))[0])[::2]
#
# print(count_array)
# print(count_array.sum()/len(count_array))
