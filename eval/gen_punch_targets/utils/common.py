import os
import random
import numpy as np
import json


def convert_from_py_space_to_unity_space(punch_targets):
    punch_targets[:, 0] = -punch_targets[:, 0]
    return punch_targets


def convert_arr_for_matplotlib(punch_targets):
    punch_targets[:, [1, 2]] = punch_targets[:, [2, 1]]
    punch_targets[:, [0, 1]] = punch_targets[:, [1, 0]]
    return punch_targets


def setup_cube_end_pts(x_range, y_range, z_range):
    xv = np.zeros((8, 3))
    x_arr = np.array([i for j in range(4) for i in x_range])
    y_arr = np.array([i for i in y_range for j in range(2)] * 2)
    z_arr = np.array([i for i in z_range for j in range(4)])
    xv[:, 0] = x_arr[:]
    xv[:, 1] = y_arr[:]
    xv[:, 2] = z_arr[:]
    return xv


# def gen_random_punch_targets(x_range, y_range, z_range, save_path, hand):
#     punch_targets = []
#     for i in range(125):
#         x = random.uniform(*x_range)
#         y = random.uniform(*y_range)
#         z = random.uniform(*z_range)
#         punch_targets.append([x, y, z])
#
#     with open(os.path.join(save_path, "punch_targets_" + hand + ".json"), 'w') as f:
#         json.dump(punch_targets, f)
#
#     xv = setup_cube_end_pts(x_range, y_range, z_range)
#
#     punch_targets = np.array(punch_targets)
#     punch_targets[:, [1, 2]] = punch_targets[:, [2, 1]]
#     punch_targets[:, [0, 1]] = punch_targets[:, [1, 0]]
#
#     xv[:, [1, 2]] = xv[:, [2, 1]]
#     xv[:, [0, 1]] = xv[:, [1, 0]]
#
#     cube_grid_display(xv, 100, np.array(punch_targets),
#                       os.path.join(save_path, "punch_targets_" + hand + ".png"))


def get_mins_maxs(punch_targets, offset=0.05):
    x, y, z = punch_targets[:, 0], punch_targets[:, 1], punch_targets[:, 2]
    minimums = [i.min() - offset for i in [x, y, z]]
    maximums = [i.max() + offset for i in [x, y, z]]
    x_r = [minimums[0], maximums[0]]
    y_r = [minimums[1], maximums[1]]
    z_r = [minimums[2], maximums[2]]
    return x_r, y_r, z_r


def cube_grid_display(xv, xg, filename, hand, plot_type):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    #
    #  Draw the grid points.
    #
    if hand == "left":
        c = 'b'
        marker = 'x'
    else:
        c = 'g'
        marker = "+"
    ax.scatter(xg[:, 0], xg[:, 1], xg[:, 2], 'b', c=c, marker=marker)
    #
    #  Outline the hexahedron by its edges.
    #
    ax.plot([xv[0, 0], xv[1, 0]], [xv[0, 1], xv[1, 1]], [xv[0, 2], xv[1, 2]], 'r')
    ax.plot([xv[0, 0], xv[2, 0]], [xv[0, 1], xv[2, 1]], [xv[0, 2], xv[2, 2]], 'r')
    ax.plot([xv[0, 0], xv[4, 0]], [xv[0, 1], xv[4, 1]], [xv[0, 2], xv[4, 2]], 'r')
    ax.plot([xv[1, 0], xv[3, 0]], [xv[1, 1], xv[3, 1]], [xv[1, 2], xv[3, 2]], 'r')
    ax.plot([xv[1, 0], xv[5, 0]], [xv[1, 1], xv[5, 1]], [xv[1, 2], xv[5, 2]], 'r')
    ax.plot([xv[2, 0], xv[3, 0]], [xv[2, 1], xv[3, 1]], [xv[2, 2], xv[3, 2]], 'r')
    ax.plot([xv[2, 0], xv[6, 0]], [xv[2, 1], xv[6, 1]], [xv[2, 2], xv[6, 2]], 'r')
    ax.plot([xv[3, 0], xv[7, 0]], [xv[3, 1], xv[7, 1]], [xv[3, 2], xv[7, 2]], 'r')
    ax.plot([xv[4, 0], xv[5, 0]], [xv[4, 1], xv[5, 1]], [xv[4, 2], xv[5, 2]], 'r')
    ax.plot([xv[4, 0], xv[6, 0]], [xv[4, 1], xv[6, 1]], [xv[4, 2], xv[6, 2]], 'r')
    ax.plot([xv[5, 0], xv[7, 0]], [xv[5, 1], xv[7, 1]], [xv[5, 2], xv[7, 2]], 'r')
    ax.plot([xv[6, 0], xv[7, 0]], [xv[6, 1], xv[7, 1]], [xv[6, 2], xv[7, 2]], 'r')

    ax.set_ylabel('X axis (lateral direction)')
    ax.set_xlabel('Z axis (forward direction)')
    ax.set_zlabel('Y axis')
    ax.set_title(plot_type.capitalize() + " " + hand.lower() + ' punch targets')
    ax.grid(True)
    # ax.axis ( 'equal' )
    plt.savefig(filename)

    fig.patch.set_visible(False)
    ax.axis('off')
    ax.title.set_visible(False)
    plt.savefig(filename.split(".png")[0]+"_no_bg.png", transparent=True)

    # with open('test.png', 'wb') as outfile:
    #     fig.canvas.print_png(outfile)

    # plt.show ( )

    plt.clf()

    return


def plot_punch_targets(punch_targets, x_range, y_range, z_range, save_path, hand, plot_type, space="unity"):
    punch_targets = np.array(punch_targets)
    if space == "unity":
        punch_targets = convert_from_py_space_to_unity_space(punch_targets)
    punch_targets = convert_arr_for_matplotlib(punch_targets)

    xv = setup_cube_end_pts(x_range, y_range, z_range)
    if space == "unity":
        xv = convert_from_py_space_to_unity_space(xv)
    xv = convert_arr_for_matplotlib(xv)

    cube_grid_display(xv, punch_targets,
                      os.path.join(save_path, plot_type + "_punch_targets_" + hand + ".png"), hand, plot_type)
