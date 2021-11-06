import numpy as np
import pandas as pd

start_r = [289, 341]
end_r = [318, 364]

start_l = [324, 368]
end_l = [340, 381]


def calc_avg_step(st, end):
    st = np.array(st)
    end = np.array(end)

    diff = end - st
    mean = np.mean(diff)
    return mean


l_mean = calc_avg_step(start_l, end_l)
r_mean = calc_avg_step(start_r, end_r)

overall_mean = np.mean([l_mean, r_mean])

# info = {
#     "right": round(r_mean, 3),
#     "left": round(l_mean, 3),
#     "overall": round(overall_mean, 3)
# }
# df = pd.DataFrame.from_dict(info, orient='index', columns=["average duration"])

step_type = ["right", "left", "overall"]
step_durations = [round(r_mean,3), round(l_mean,3), round(overall_mean,3)]

info = {
    "step type": step_type,
    "average": step_durations,
}
df = pd.DataFrame(info)
print(df.to_latex(index=False))
