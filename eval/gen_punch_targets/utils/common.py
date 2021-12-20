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
    plt.figure(dpi=1200)
    # x1, x2, y1, y2 = plt.axis()
    # plt.axis((x1, x2, y1 - 100, y2 + 100))
    # plt.style.use('dark_background')
    fig = plt.figure()
    # fig.set_tight_layout(True)
    fig.patch.set_facecolor('darkgray')
    ax = fig.add_subplot(111, projection='3d')
    x_scale = 1.25
    y_scale = 1
    z_scale = 1
    # x_scale = 1.25
    # y_scale = 1.15
    # z_scale = 1

    scale = np.diag([x_scale, y_scale, z_scale, 1.0])
    scale = scale * (1.0 / scale.max())
    scale[3, 3] = 1.0

    def short_proj():
        return np.dot(Axes3D.get_proj(ax), scale)

    ax.get_proj = short_proj
    # ax.set_facecolor('darkgray')
    #
    #  Draw the grid points.
    #
    if hand == "left":
        # c = 'cyan'
        c = "blue"
        marker = 'x'
        s = 50 / 5
        # s=2
    else:
        # c = 'yellow'
        c = 'orange'
        marker = "+"
        s = 70 / 5
        # s = 3
    # s = [10 * 4 ** n for n in range(len(xg[:, 0].ravel()))]
    ax.scatter(xg[:, 0], xg[:, 1], xg[:, 2], 'b', c=c, marker=marker, s=s)
    #
    #  Outline the hexahedron by its edges.
    #
    # ax.plot([xv[0, 0], xv[1, 0]], [xv[0, 1], xv[1, 1]], [xv[0, 2], xv[1, 2]], 'white')
    # ax.plot([xv[0, 0], xv[2, 0]], [xv[0, 1], xv[2, 1]], [xv[0, 2], xv[2, 2]], 'white')
    # ax.plot([xv[0, 0], xv[4, 0]], [xv[0, 1], xv[4, 1]], [xv[0, 2], xv[4, 2]], 'white')
    # ax.plot([xv[1, 0], xv[3, 0]], [xv[1, 1], xv[3, 1]], [xv[1, 2], xv[3, 2]], 'white')
    # ax.plot([xv[1, 0], xv[5, 0]], [xv[1, 1], xv[5, 1]], [xv[1, 2], xv[5, 2]], 'white')
    # ax.plot([xv[2, 0], xv[3, 0]], [xv[2, 1], xv[3, 1]], [xv[2, 2], xv[3, 2]], 'white')
    # ax.plot([xv[2, 0], xv[6, 0]], [xv[2, 1], xv[6, 1]], [xv[2, 2], xv[6, 2]], 'white')
    # ax.plot([xv[3, 0], xv[7, 0]], [xv[3, 1], xv[7, 1]], [xv[3, 2], xv[7, 2]], 'white')
    # ax.plot([xv[4, 0], xv[5, 0]], [xv[4, 1], xv[5, 1]], [xv[4, 2], xv[5, 2]], 'white')
    # ax.plot([xv[4, 0], xv[6, 0]], [xv[4, 1], xv[6, 1]], [xv[4, 2], xv[6, 2]], 'white')
    # ax.plot([xv[5, 0], xv[7, 0]], [xv[5, 1], xv[7, 1]], [xv[5, 2], xv[7, 2]], 'white')
    # ax.plot([xv[6, 0], xv[7, 0]], [xv[6, 1], xv[7, 1]], [xv[6, 2], xv[7, 2]], 'white')

    ax.plot([xv[0, 0], xv[1, 0]], [xv[0, 1], xv[1, 1]], [xv[0, 2], xv[1, 2]], 'black')
    ax.plot([xv[0, 0], xv[2, 0]], [xv[0, 1], xv[2, 1]], [xv[0, 2], xv[2, 2]], 'black')
    ax.plot([xv[0, 0], xv[4, 0]], [xv[0, 1], xv[4, 1]], [xv[0, 2], xv[4, 2]], 'black')
    ax.plot([xv[1, 0], xv[3, 0]], [xv[1, 1], xv[3, 1]], [xv[1, 2], xv[3, 2]], 'black')
    ax.plot([xv[1, 0], xv[5, 0]], [xv[1, 1], xv[5, 1]], [xv[1, 2], xv[5, 2]], 'black')
    ax.plot([xv[2, 0], xv[3, 0]], [xv[2, 1], xv[3, 1]], [xv[2, 2], xv[3, 2]], 'black')
    ax.plot([xv[2, 0], xv[6, 0]], [xv[2, 1], xv[6, 1]], [xv[2, 2], xv[6, 2]], 'black')
    ax.plot([xv[3, 0], xv[7, 0]], [xv[3, 1], xv[7, 1]], [xv[3, 2], xv[7, 2]], 'black')
    ax.plot([xv[4, 0], xv[5, 0]], [xv[4, 1], xv[5, 1]], [xv[4, 2], xv[5, 2]], 'black')
    ax.plot([xv[4, 0], xv[6, 0]], [xv[4, 1], xv[6, 1]], [xv[4, 2], xv[6, 2]], 'black')
    ax.plot([xv[5, 0], xv[7, 0]], [xv[5, 1], xv[7, 1]], [xv[5, 2], xv[7, 2]], 'black')
    ax.plot([xv[6, 0], xv[7, 0]], [xv[6, 1], xv[7, 1]], [xv[6, 2], xv[7, 2]], 'black')

    ax.set_ylim(-0.6, 0.4)
    # ax.set_ylim(0.4, 1.6)
    ax.set_xlim(0.15, 0.95)
    ax.set_zlim(1, 2)

    # ax.set_ylim(-0.6, 0.4)
    # ax.set_xlim(0.25, 0.8)
    # ax.set_zlim(1.1, 1.7)

    ax.set_ylabel('X axis (lateral direction) in m')
    ax.set_xlabel('Z axis (forward direction) in m')
    ax.set_zlabel('Y axis in m')
    ax.set_title(plot_type.capitalize() + " " + hand.lower() + ' punch targets')
    ax.grid(False)
    # ax.axis ( 'equal' )

    plt.gca().invert_xaxis()
    # plt.gca().invert_yaxis()

    plt.savefig(filename, dpi=500)
    # ,bbox_inches="tight")

    fig.patch.set_visible(False)
    ax.axis('off')
    ax.title.set_visible(False)
    plt.savefig(filename.split(".png")[0] + "_no_bg.png", transparent=True, dpi=500)

    # with open('test.png', 'wb') as outfile:
    #     fig.canvas.print_png(outfile)

    # plt.show ( )

    plt.clf()

    return


def cube_grid_display_colored(xv, xg, target_colors, target_transparency,filename, hand, plot_type):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    plt.figure(dpi=1200)
    # x1, x2, y1, y2 = plt.axis()
    # plt.axis((x1, x2, y1 - 100, y2 + 100))
    # plt.style.use('dark_background')
    fig = plt.figure()
    # fig.set_tight_layout(True)
    fig.patch.set_facecolor('darkgray')
    ax = fig.add_subplot(111, projection='3d')
    x_scale = 1.25
    y_scale = 1
    z_scale = 1
    # x_scale = 1.25
    # y_scale = 1.15
    # z_scale = 1

    scale = np.diag([x_scale, y_scale, z_scale, 1.0])
    scale = scale * (1.0 / scale.max())
    scale[3, 3] = 1.0

    def short_proj():
        return np.dot(Axes3D.get_proj(ax), scale)

    ax.get_proj = short_proj
    # ax.set_facecolor('darkgray')
    #
    #  Draw the grid points.
    #
    if hand == "left":
        # c = 'cyan'
        c = "blue"
        marker = 'x'
        s = 50 / 5
        # s=2
    else:
        # c = 'yellow'
        c = 'orange'
        marker = "+"
        s = 70 / 5
        # s = 3
    # s = [10 * 4 ** n for n in range(len(xg[:, 0].ravel()))]
    ax.scatter(xg[:, 0], xg[:, 1], xg[:, 2], 'b', c=target_colors, marker=marker, s=target_transparency, alpha=0.5)
    # ax.scatter(xg[:, 0], xg[:, 1], xg[:, 2], 'b', c=target_colors, marker=marker, s=s)
    #
    #  Outline the hexahedron by its edges.
    #
    # ax.plot([xv[0, 0], xv[1, 0]], [xv[0, 1], xv[1, 1]], [xv[0, 2], xv[1, 2]], 'white')
    # ax.plot([xv[0, 0], xv[2, 0]], [xv[0, 1], xv[2, 1]], [xv[0, 2], xv[2, 2]], 'white')
    # ax.plot([xv[0, 0], xv[4, 0]], [xv[0, 1], xv[4, 1]], [xv[0, 2], xv[4, 2]], 'white')
    # ax.plot([xv[1, 0], xv[3, 0]], [xv[1, 1], xv[3, 1]], [xv[1, 2], xv[3, 2]], 'white')
    # ax.plot([xv[1, 0], xv[5, 0]], [xv[1, 1], xv[5, 1]], [xv[1, 2], xv[5, 2]], 'white')
    # ax.plot([xv[2, 0], xv[3, 0]], [xv[2, 1], xv[3, 1]], [xv[2, 2], xv[3, 2]], 'white')
    # ax.plot([xv[2, 0], xv[6, 0]], [xv[2, 1], xv[6, 1]], [xv[2, 2], xv[6, 2]], 'white')
    # ax.plot([xv[3, 0], xv[7, 0]], [xv[3, 1], xv[7, 1]], [xv[3, 2], xv[7, 2]], 'white')
    # ax.plot([xv[4, 0], xv[5, 0]], [xv[4, 1], xv[5, 1]], [xv[4, 2], xv[5, 2]], 'white')
    # ax.plot([xv[4, 0], xv[6, 0]], [xv[4, 1], xv[6, 1]], [xv[4, 2], xv[6, 2]], 'white')
    # ax.plot([xv[5, 0], xv[7, 0]], [xv[5, 1], xv[7, 1]], [xv[5, 2], xv[7, 2]], 'white')
    # ax.plot([xv[6, 0], xv[7, 0]], [xv[6, 1], xv[7, 1]], [xv[6, 2], xv[7, 2]], 'white')

    ax.plot([xv[0, 0], xv[1, 0]], [xv[0, 1], xv[1, 1]], [xv[0, 2], xv[1, 2]], 'black')
    ax.plot([xv[0, 0], xv[2, 0]], [xv[0, 1], xv[2, 1]], [xv[0, 2], xv[2, 2]], 'black')
    ax.plot([xv[0, 0], xv[4, 0]], [xv[0, 1], xv[4, 1]], [xv[0, 2], xv[4, 2]], 'black')
    ax.plot([xv[1, 0], xv[3, 0]], [xv[1, 1], xv[3, 1]], [xv[1, 2], xv[3, 2]], 'black')
    ax.plot([xv[1, 0], xv[5, 0]], [xv[1, 1], xv[5, 1]], [xv[1, 2], xv[5, 2]], 'black')
    ax.plot([xv[2, 0], xv[3, 0]], [xv[2, 1], xv[3, 1]], [xv[2, 2], xv[3, 2]], 'black')
    ax.plot([xv[2, 0], xv[6, 0]], [xv[2, 1], xv[6, 1]], [xv[2, 2], xv[6, 2]], 'black')
    ax.plot([xv[3, 0], xv[7, 0]], [xv[3, 1], xv[7, 1]], [xv[3, 2], xv[7, 2]], 'black')
    ax.plot([xv[4, 0], xv[5, 0]], [xv[4, 1], xv[5, 1]], [xv[4, 2], xv[5, 2]], 'black')
    ax.plot([xv[4, 0], xv[6, 0]], [xv[4, 1], xv[6, 1]], [xv[4, 2], xv[6, 2]], 'black')
    ax.plot([xv[5, 0], xv[7, 0]], [xv[5, 1], xv[7, 1]], [xv[5, 2], xv[7, 2]], 'black')
    ax.plot([xv[6, 0], xv[7, 0]], [xv[6, 1], xv[7, 1]], [xv[6, 2], xv[7, 2]], 'black')

    ax.set_ylim(-0.6, 0.4)
    # ax.set_ylim(0.4, 1.6)
    ax.set_xlim(0.15, 0.95)
    ax.set_zlim(1, 2)

    # ax.set_ylim(-0.6, 0.4)
    # ax.set_xlim(0.25, 0.8)
    # ax.set_zlim(1.1, 1.7)

    ax.set_ylabel('X axis (lateral direction) in m')
    ax.set_xlabel('Z axis (forward direction) in m')
    ax.set_zlabel('Y axis in m')
    ax.set_title(plot_type.capitalize() + " " + hand.lower() + ' punch targets')
    ax.grid(False)
    # ax.axis ( 'equal' )

    plt.gca().invert_xaxis()
    # plt.gca().invert_yaxis()

    plt.savefig(filename, dpi=500)
    # ,bbox_inches="tight")

    fig.patch.set_visible(False)
    ax.axis('off')
    ax.title.set_visible(False)
    plt.savefig(filename.split(".png")[0] + "_no_bg.png", transparent=True, dpi=500)

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


def plot_colored_punch_targets(punch_targets, target_colors, target_transparency, x_range, y_range, z_range,
                               save_path, hand, plot_type, space="unity"):
    punch_targets = np.array(punch_targets)
    if space == "unity":
        punch_targets = convert_from_py_space_to_unity_space(punch_targets)
    punch_targets = convert_arr_for_matplotlib(punch_targets)

    xv = setup_cube_end_pts(x_range, y_range, z_range)
    if space == "unity":
        xv = convert_from_py_space_to_unity_space(xv)
    xv = convert_arr_for_matplotlib(xv)

    cube_grid_display_colored(xv, punch_targets, target_colors, target_transparency,
                      os.path.join(save_path, plot_type + "_punch_targets_" + hand + ".png"), hand, plot_type)
