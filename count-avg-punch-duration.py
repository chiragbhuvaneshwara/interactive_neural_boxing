import os

import numpy as np
import pandas as pd
from scipy import stats

# binary_punch_phase_df = pd.read_csv("Blender Code Snippets/data annotation res/test3.csv")
# binary_punch_phase_df = pd.read_csv("Blender Code Snippets/data annotation res/new_data/boxing_2_binary.csv")
# binary_punch_phase_df = pd.read_csv("Blender Code Snippets/data annotation res/new_data/boxing_2_tertiary.csv")
# print(binary_punch_phase_df)

# condition = np.array([True, True, True, False, False, True, True, False, True, True, True, True, False, False, True])
# condition = np.array([1,1,0,1,1,1,1,0,1])


def get_punch_count(df, hand, punch_val):
    condition = np.array(df[hand] == punch_val)

    match_val = True
    count_array = np.diff(np.where(np.concatenate(([condition[0]],
                                                   condition[:-1] != condition[1:],
                                                   [match_val])))[0])[::2]

    print('hand: ', hand, ',dir: ', punch_val)
    # print('count: ', count_array)
    print('len of count: ', len(count_array))
    # print('mean: ', count_array.mean())
    # print('median: ', np.median(count_array))

    # unique_elements, counts_elements = np.unique(count_array, return_counts=True)
    # print("Frequency of unique values:")
    # el_count = {k: v for k, v in zip(unique_elements, counts_elements)}
    # print(el_count)
    # print(np.asarray((unique_elements, counts_elements)))

    print('\n')
    return count_array


# data_dir = "Blender Code Snippets/data annotation res/new_data/tertiary"
data_dir = "Blender Code Snippets/data annotation res/new_data/binary"
countrf = np.array([])
countrr = np.array([])
countlf = np.array([])
countlr = np.array([])
for file in os.listdir(data_dir):
    filename = os.path.join(data_dir, file)
    punch_phase_df = pd.read_csv(filename)
    countrf = np.append(countrf, get_punch_count(punch_phase_df, 'right punch', 1.0))
    # countrr = np.append(countrr, get_punch_count(punch_phase_df, 'right punch', -1.0))
    print('###############################################################################')
    countlf = np.append(countlf, get_punch_count(punch_phase_df, 'left punch', 1.0))
    # countlr = np.append(countlr, get_punch_count(punch_phase_df, 'left punch', -1.0))
    print('******************************************************************************')

# for i in [countrf, countrr, countlf, countlr]:
for i in [countrf, countlf]:
    i = np.array(i)
    print(len(i), i.mean())

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
