import pandas as pd

punch_phase_csv_path = 'C:/Users/chira/OneDrive/Documents/Uni/Thesis/VCS-boxing-predictor/Blender Code Snippets/data annotation res/PunchPhase_detailed.csv'
frame_rate_offset = 2
frame_rate_divisor = 1
punch_phase_df = pd.read_csv(punch_phase_csv_path)
print(punch_phase_df.values.shape)
print(punch_phase_df.columns)
this = punch_phase_df['Unnamed: 0'].to_numpy()[-1]
# print(len(this))
print(this)

total_frames = len(punch_phase_df.index)
rows_to_keep = [i for i in range(frame_rate_offset, total_frames, frame_rate_divisor)]

punch_phase_df = punch_phase_df.iloc[rows_to_keep, :]
punch_phase_df = punch_phase_df.loc[:, ~punch_phase_df.columns.str.contains('^Unnamed')]

# convert dataframe to numpy array
punch_phase = punch_phase_df.values
# print(punch_phase[:2])

# punch_phase = punch_phase_df.to_numpy()
# # deleting row and column indices
# punch_phase = np.delete(punch_phase, 0, axis=1)
# punch_phase = np.delete(punch_phase, 0, axis=0)

punch_dphase = punch_phase[1:] - punch_phase[:-1]
punch_dphase[punch_dphase < 0] = (1.0 - punch_phase[:-1] + punch_phase[1:])[punch_dphase < 0]