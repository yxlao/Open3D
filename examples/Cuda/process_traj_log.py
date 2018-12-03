import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def save_traj_log(log_path, trajectory): # n x 4 x 4
    with open(log_path, 'w') as fout:
        for i in range(0, trajectory.shape[0]):
            T = trajectory[i, :, :]
            fout.write('{} {} {}\n'.format(i, i, i + 1))

            for r in range(0, 4):
                for h in range(0, 4):
                    fout.write('{} '.format(T[r, h]))
                fout.write('\n')


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

    trajectory_ic = load_traj_log(
        '../../cmake-build-release/bin/examples/trajectory_ic.log')
    trajectory_fc = load_traj_log(
        '../../cmake-build-release/bin/examples/trajectory_fc.log')
    trajectory_gt = load_traj_log(
        '../../cmake-build-release/bin/examples/trajectory_gt.log')

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    fig = plt.figure()
    #ax = fig.add_subplot(111)
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(trajectory_fc[:, 0], trajectory_fc[:, 1], trajectory_fc[:, 2],
            'g', label=r'FC')
    ax.plot(trajectory_ic[:, 0], trajectory_ic[:, 1], trajectory_ic[:, 2],
            'b', label=r'IC')
    ax.plot(trajectory_gt[:, 0], trajectory_gt[:, 1], trajectory_gt[:, 2],
            'r', label=r'GT')


    ax.set_xlabel(r'x', fontsize=16)
    ax.set_ylabel(r'y', fontsize=16)
    # ax.set_zlabel('z')
    plt.legend(fontsize=14)
    plt.show()
