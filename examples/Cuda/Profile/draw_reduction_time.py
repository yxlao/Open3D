import matplotlib.pyplot as plt
import numpy as np


def draw_text(ax, x, y):
    max_y = np.max(y)
    for i, v in enumerate(y):
        text = str(y[i])
        ax.text(x[i] - 0.045 * len(text), y[i],
                r'\textbf{' + text + '}')


def draw_error_bar(ax, x, xticks, y_gpu, e_gpu, y_cpu, e_cpu, title, ylabel):
    offset = 0.2
    width = offset * 2

    bar_cpu = ax.bar(x, y_cpu, width=width, color=(0.26, 0.52, 0.96))
    ax.errorbar(x + offset, y_cpu, yerr=e_cpu, fmt='.r', capsize=10)
    # draw_text(ax, x + offset, y_cpu, offset)

    bar_gpu = ax.bar(x - offset, y_gpu, width=width, color=(0.86, 0.27, 0.22))
    ax.errorbar(x - offset, y_gpu, yerr=e_gpu, fmt='.b', capsize=10)
    # draw_text(ax, x - offset, y_gpu)

    ax.yaxis.grid(True)
    ax.legend((bar_gpu, bar_cpu), ('GPU', 'CPU'))
    ax.set_xticks(x)
    ax.set_xticklabels(xticks)
    ax.set_title(title)
    ax.set_ylabel(ylabel)


if __name__ == '__main__':
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    fig, axs = plt.subplots(figsize=(4, 3))

    x = np.array([1])
    labels = [r'\textit{float}']# [r'\textit{int}', r'\textit{float}']

    # y_add = [0.1869, 0.4372]
    # y_sfl = [0.3253, 0.8409]
    # y_hrs = [0.2397, 0.2405]
    y_add = [0.4372]
    y_sfl = [0.8409]
    y_hrs = [0.2405]


    offset = 0.25
    width = offset
    bar_add = axs.bar(x - offset, y_add, width=width, color=(0.26, 0.52, 0.96))
    bar_sfl = axs.bar(x, y_sfl, width=width, color=(0.86, 0.27, 0.22))
    bar_hrs = axs.bar(x + offset, y_hrs, width=width, color=(0.96, 0.71, 0))

    axs.yaxis.grid(True)
    axs.legend((bar_add, bar_sfl, bar_hrs), (r'atomicAdd', r'warp shuffle',
                                             r'original'))
    axs.set_xticks(x)
    axs.set_xticklabels(labels)
    axs.set_title(r'\textbf{Summation on a 640x480 image}')
    axs.set_ylabel(r'\textit{Time} (ms)')
    fig.tight_layout()

    plt.savefig('reduction.pdf', bbox_inches='tight')
    plt.show()

