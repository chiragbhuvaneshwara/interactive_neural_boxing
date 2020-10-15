from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np


class Plotter():
    """
    This class helps to plot a point-cloud (e.g. joint positions) in order to visualize the result.

    @author: Janis Sprenger
    """

    def __init__(self):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        # self.ax.view_init(30, 90)

    def plot(self, pose, axis=[]):
        """
        Plot can be used to plot a single frame of the animation.

        Arguments:
            pose {np.array(n_joints, 3)} -- joint positions

        Keyword Arguments:
            axis {np.array(n_joints, 1-3, 3)} -- opt. axis definitions for each joint (default: {[]})
        """
        if len(axis) == 0:
            Plotter.update_animation(0, self.ax, [pose], [])
        else:
            Plotter.update_animation(0, self.ax, [pose], [axis])
        plt.show()

    def animated(self, poses, axis=[]):
        """
        Animated can be utilized to animate a set of poses.

        Arguments:
            poses {np.array(n_frames, n_joints, 3)} -- joint positions for n_frames.

        Keyword Arguments:
            axis {np.array(n_frames, n_joints, 1-3, 3)} -- opt. axis definitions for each joint and frame (default: {[]})
        """
        N = len(poses)
        anim = animation.FuncAnimation(self.fig, Plotter.update_animation, N, fargs=(self.ax, poses, axis), interval=100/N)
        plt.show()


    @staticmethod
    def update_animation(num_pose, ax, poses, axes=[]):
        poses = np.array(poses)
        axes = np.array(axes)
        color = ["r", "g", "b"]
        ax.clear()
        pose = poses[num_pose]

        p = pose
        # p = np.array(pose, dtype=np.float).T
        # p[1,:], p[2,:] = p[2,:], p[1,:]
        ax.scatter([0], [0], [0], marker="x")

        ax.plot([0, 1], [0, 0], zs=[0, 0], c=color[0])
        ax.plot([0, 0], [0, 1], zs=[0, 0], c=color[1])
        ax.plot([0, 0], [0, 0], zs=[0, 1], c=color[2])

        ax.scatter(p[:, 0], p[:, 1], p[:, 2], marker="o")
        for i in range(len(p)):
            ax.text(p[i, 0], p[i, 1], p[i, 2], i)

        if len(axes) > 0:
            axis = np.array(axes[num_pose], dtype=np.float)
            for i in range(axis.shape[1]):
                aux_x = 0.2 * axis[:, i, 0] + p[:, 0]
                aux_y = 0.2 * axis[:, i, 1] + p[:, 1]
                aux_z = 0.2 * axis[:, i, 2] + p[:, 2]

                aux_x_2 = 0.2 * axis[:, i, 0] + p[16, 0]
                aux_y_2 = 0.2 * axis[:, i, 1] + p[16, 1]
                aux_z_2 = 0.2 * axis[:, i, 2] + p[16, 2]
                ax.scatter(aux_x, aux_y, aux_z, marker=".", c=color[i])
                ax.scatter(aux_x_2, aux_y_2, aux_z_2, marker="*", c=color[i])
                for j in range(len(aux_x)):
                    ax.plot([p[0, j], aux_x[j]], [p[1, j], aux_y[j]], zs=[p[2, j], aux_z[j]], c=color[i])
                    if j == 16:
                        ax.plot([p[0, j], aux_x_2[j]], [p[1, j], aux_y_2[j]], zs=[p[2, j], aux_z_2[j]], c=color[i])

        span_x = np.max(p[:, 0]) - np.min(p[:, 0])
        span_y = np.max(p[:, 1]) - np.min(p[:, 1])
        span_z = np.max(p[:, 2]) - np.min(p[:, 2])

        span = max(span_x, max(span_y, span_z))

        ax.set_xlim3d([np.min(p[:, 0]) - 0.5 * span, np.min(p[:, 0]) + 0.5 * span])
        ax.set_ylim3d([np.min(p[:, 1]) - 0.5 * span, np.min(p[:, 1]) + 0.5 * span])
        ax.set_zlim3d([np.min(p[:, 2]), np.min(p[:, 2]) + span])
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')
        return

