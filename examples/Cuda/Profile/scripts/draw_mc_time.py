import matplotlib.pyplot as plt
import numpy as np


def draw_text(ax, x, y):
    max_y = np.max(y)
    for i, v in enumerate(y):
        text = str(y[i])
        ax.text(x[i] - 0.045 * len(text), y[i],
                r'\textbf{' + text + '}')


def draw_error_bar(ax, x, xticks, y_gpu, e_gpu, title, ylabel):
    offset = 0.2
    width = offset * 2

    bar_gpu = ax.bar(x, y_gpu, width=width, color=(0.86, 0.27, 0.22))
    ax.errorbar(x, y_gpu, yerr=e_gpu, fmt='.', color=(0.96, 0.71, 0),capsize=10)

    ax.yaxis.grid(True)
    ax.set_xticks(x)
    ax.set_xticklabels(xticks)
    ax.set_title(title)
    ax.set_ylabel(ylabel)


if __name__ == '__main__':
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    fig, axs = plt.subplots(figsize=(5, 3))
    x = np.array([1, 2, 3, 4, 5])
    labels = [r'\textit{fr2\_desktop}', r'\textit{fr3\_household}',
              r'\textit{lounge}', r'\textit{copyroom}',
              r'\textit{livingroom1}']
# mean = 8.713374, std = 1.637024

    y_gpu_mc = np.array([8.71, 11.63, 6.28, 5.18, 4.07])
    e_gpu_mc = np.array([1.64, 3.25, 1.98, 1.94, 1.15])
    draw_error_bar(axs, x, labels,
                   y_gpu_mc, e_gpu_mc,
                   r'\textbf{Marching Cubes for Voxels in Frustum}',
                   r'\textbf{Average time per frame} (ms)')

    fig.tight_layout()

    # bar_cpu_odom = plt.bar(x + offset, y_cpu_odom, width=width)
    # plt.errorbar(x + offset, y_cpu_odom, yerr=e_cpu_odom, fmt='.g', capsize=20)
    # for i, v in enumerate(y_cpu_odom):
    #     plt.text(x[i] + offset + 0.02, y_cpu_odom[i] + 5, str(y_cpu_odom[i]))

    plt.savefig('mc_time.pdf', bbox_inches='tight')
    plt.show()

