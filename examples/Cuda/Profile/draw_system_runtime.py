import matplotlib.pyplot as plt
import numpy as np


def draw_text(ax, x, y):
    max_y = np.max(y)
    for i, v in enumerate(y):
        text = str(y[i])
        ax.text(x[i] - 0.045 * len(text), y[i],
                r'\textbf{' + text + '}')


def draw_error_bar(ax, x, xticks,
                   y_make, y_reg, y_refine, y_int,
                   title, ylabel):
    offset = 0.2
    width = offset

    bar_make = ax.bar(x - 1.5 * offset, y_make,
                      width=width, color=(0.26, 0.52, 0.96))
    bar_reg = ax.bar(x - 0.5 * offset, y_reg,
                     width=width, color=(0.86, 0.27, 0.22))
    bar_refine = ax.bar(x + 0.5 * offset, y_refine,
                     width=width, color=(0.96, 0.71, 0))
    bar_int = ax.bar(x + 1.5 * offset, y_int,
                    width=width, color=(0.05, 0.62, 0.35))

    ax.yaxis.grid(True)
    ax.legend((bar_make, bar_reg, bar_refine, bar_int),
              ('make submaps', 'register submaps', 'refine registration',
               'integration'), loc=0)
    ax.set_xticks(x)
    ax.set_xticklabels(xticks)
    ax.set_title(title)
    ax.set_ylabel(ylabel)


if __name__ == '__main__':
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    fig, axs = plt.subplots(figsize=(12, 3))
    x = np.array([1, 2, 3, 4, 5])
    labels = [r'\textit{fr2\_desktop} (2893 frames)',
              r'\textit{fr3\_household} (2488 frames)',
              r'\textit{lounge} (3000 frames)',
              r'\textit{copyroom} (5490 frames)',
              r'\textit{livingroom1} (2870 frames)']

    y_make = np.array([161.85, 139.60, 160.19, 281.18, 120.01])
    y_reg = np.array([29.97, 21.14, 37.08, 91.01, 120.40])
    y_refine = np.array([321.03, 124.43, 155.91, 256.19, 221.06])
    y_int = np.array([69.77, 62.17, 58.69, 99.78, 37.01])

    draw_error_bar(axs, x, labels,
                   y_make, y_reg, y_refine, y_int,
                   r'\textbf{Total Runtime of Components (w/o local loop '
                   r'closures)}',
                   r'\textbf{Time} (s)')

    fig.tight_layout()

    # bar_cpu_odom = plt.bar(x + offset, y_cpu_odom, width=width)
    # plt.errorbar(x + offset, y_cpu_odom, yerr=e_cpu_odom, fmt='.g', capsize=20)
    # for i, v in enumerate(y_cpu_odom):
    #     plt.text(x[i] + offset + 0.02, y_cpu_odom[i] + 5, str(y_cpu_odom[i]))

    plt.savefig('total_time.pdf', bbox_inches='tight')
    plt.show()

