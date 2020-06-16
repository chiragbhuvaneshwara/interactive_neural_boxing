import numpy as np
import pandas as pd

def load_blender_data(blender_csv_file_path, frame_rate_divisor=2, frame_rate_offset=0):
    # print('Processing Blender csv data %s' % punch_phase_csv_path, frame_rate_divisor, frame_rate_offset)

    blender_df = pd.read_csv(blender_csv_file_path, index_col=0)
    # blender_df.drop([0], axis=1)

    reqd = blender_df['right punch'] > 0
    # print(reqd.astype(int))

    blender_df_one_hot = (blender_df > 0) * 1
    blender_df_one_hot.to_csv('onehot.csv')

    blender_df_tertiary = blender_df.apply(lambda x: x.is_monotonic_increasing)
    print(blender_df_tertiary)

    def monotonic(x):
        # return np.all(np.diff(x) > 0)
        return np.diff(x) > 0

    b = monotonic(blender_df['right punch'].values)
    print(b)

    total_frames = len(blender_df.index)
    # rows_to_keep = [i for i in range(frame_rate_offset, total_frames, frame_rate_divisor)]
    #
    # blender_df = blender_df.iloc[rows_to_keep, :]

    # convert dataframe to numpy array
    # blender_np = blender_df.to_numpy()
    # blender_np = np.delete(blender_np, 0, axis=1)
    # blender_np = np.delete(blender_np, 0, axis=0)
    # print('-------------------------')
    # print(self.n_frames)
    # print(blender_np[5600,:])

    # return blender_np

load_blender_data('C:/Users/chira/OneDrive/Documents/Uni/Thesis/VCS-boxing-predictor/Blender Code Snippets/data annotation res/Punch.csv')