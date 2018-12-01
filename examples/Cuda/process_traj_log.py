import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def load_traj_log(log_path):

    with open(log_path, 'r') as fin:
        lines = fin.read().splitlines()

        positions = []

        for i in range(0, len(lines), 5):
            T1 = lines[i + 1].split(' ')
            T2 = lines[i + 2].split(' ')
            T3 = lines[i + 3].split(' ')

            positions.append(list(map(float, [T1[3], T2[3], T3[3]])))

        return np.stack(positions)


if __name__ == '__main__':

    plt.rc('text', usetex=False)
    plt.rc('font', family='serif')
    #
    trajectory = load_traj_log(
        '../../cmake-build-release/bin/examples/trajectory.log')

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()
    #
    # fig, ax = plt.subplots(1, 1)
    #
    # nframes, niters = losses_0.shape
    # for i in range(nframes):
    #     ax.plot(np.arange(niters), np.log(losses_0[i, :]), linewidth=1)
    #
    # ax.grid()
    # ax.set_xlabel(r'\textbf{Iterations}')
    # ax.set_xticks(np.arange(0, niters, 2))
    # ax.set_ylabel(r'\textit{Loss}', fontsize=16)
    # ax.set_title(r"FC Odometry", fontsize=16, color='gray')
    #
    # # Make room for the ridiculously large title.
    # plt.subplots_adjust(top=0.8)
    # plt.savefig('fc.png')
    # plt.show()
