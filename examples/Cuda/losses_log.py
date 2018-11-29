import numpy as np
import matplotlib.pyplot as plt


def load_losses_log(log_path):
    with open(log_path, 'r') as fin:
        lines = fin.read().splitlines()

        # We assume fixed 3 level pyramids
        losses_0 = []
        losses_1 = []
        losses_2 = []
        for i in range(0, len(lines), 4):
            losses_0.append([float(str_number)
                             for str_number in lines[i + 1].split(' ')[:-1]])
            losses_1.append([float(str_number)
                             for str_number in lines[i + 2].split(' ')[:-1]])
            losses_2.append([float(str_number)
                             for str_number in lines[i + 3].split(' ')[:-1]])

        return np.stack(losses_0), np.stack(losses_1), np.stack(losses_2)


if __name__ == '__main__':
    losses_0, losses_1, losses_2 = load_losses_log(
        '../../cmake-build-release/bin/examples/odometry-step-1.log')

    fig, ax = plt.subplots(1, 1)

    nframes, n = losses_1.shape
    for i in range(nframes):
        ax.plot(np.arange(n), np.log(losses_1[i, :]))

    plt.show()
