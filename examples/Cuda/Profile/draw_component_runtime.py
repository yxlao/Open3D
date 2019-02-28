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

    bar_cpu = ax.bar(x + offset, y_cpu, width=width, color=(0.26, 0.52, 0.96))
    ax.errorbar(x + offset, y_cpu, yerr=e_cpu, fmt='.', color=(0.96, 0.71,0), capsize=10)
    # draw_text(ax, x + offset, y_cpu, offset)

    bar_gpu = ax.bar(x - offset, y_gpu, width=width, color=(0.86, 0.27, 0.22))
    ax.errorbar(x - offset, y_gpu, yerr=e_gpu, fmt='.', color=(0.96, 0.71, 0),capsize=10)
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

    fig, axs = plt.subplots(2, 3, figsize=(16,8))
    x = np.array([1, 2, 3, 4, 5])
    labels = [r'\textit{fr2\_desktop}', r'\textit{fr3\_household}',
              r'\textit{lounge}', r'\textit{copyroom}',
              r'\textit{livingroom1}']

    y_gpu_odom = np.array([16.34, 17.20, 17.24, 16.54, 17.15])
    e_gpu_odom = np.array([3.05, 4.18, 4.24, 2.40, 3.30])
    y_cpu_odom = np.array([543.42, 561.07, 551.26, 539.58, 568.89])
    e_cpu_odom = np.array([76.43, 73.98, 64.09, 26.81, 26.35])

    y_gpu_icp = np.array([1318.05, 1455.27, 1244.91, 849.50, 750.08])
    e_gpu_icp = np.array([247.90, 436.10, 547.47, 313.63, 490.92])
    y_cpu_icp = np.array([2549.02, 2265.17, 2087.76, 1349.31, 1581.73])
    e_cpu_icp = np.array([697.25, 903.08, 1016.48, 551.72, 883.10])

    y_gpu_int = np.array([10.55, 11.10, 6.87, 5.71, 5.27])
    e_gpu_int = np.array([2.45, 2.96, 1.63, 1.51, 0.97])

    y_cpu_int = np.array([950.39, 848.95, 588.82, 539.01, 1397.97])
    e_cpu_int = np.array([277.17, 389.40, 218.23, 221.90, 677.00])

    y_gpu_fgr = np.array([84.47, 85.24, 104.20, 83.23, 81.23])
    e_gpu_fgr = np.array([14.43, 22.73, 30.62, 20.51, 18.88])
    y_cpu_fgr = np.array([444.88, 432.71, 481.15, 307.64, 293.09])
    e_cpu_fgr = np.array([95.02, 158.41, 170.68, 94.50, 77.51])

    y_gpu_fe = np.array([26.82, 27.43, 33.54, 26.49, 25.51])
    e_gpu_fe = np.array([3.94, 7.08, 9.73, 6.80, 6.21])
    y_cpu_fe = np.array([94.87, 91.04, 112.85, 79.47, 85.63])
    e_cpu_fe = np.array([15.56, 20.32, 28.17, 16.34, 18.79])

    y_gpu_fm = np.array([5.70, 5.37, 7.80, 4.47, 5.55])
    e_gpu_fm = np.array([1.35, 2.23, 4.11, 1.61, 2.18])
    y_cpu_fm = np.array([113.25, 113.29, 110.42, 66.02, 51.33])
    e_cpu_fm = np.array([31.61, 57.77, 56.01, 31.09, 18.81])

    draw_error_bar(axs[0, 0], x, labels,
                   y_gpu_odom, e_gpu_odom, y_cpu_odom, e_cpu_odom,
                   r'\textbf{Multi-scale RGBD Odometry}',
                   r'\textbf{Average time per frame pair} (ms)')
    draw_error_bar(axs[0, 1], x, labels,
                   y_gpu_int, e_gpu_int, y_cpu_int, e_cpu_int,
                   r'\textbf{TSDF Integration}',
                   r'\textbf{Average time per frame} (ms)')
    draw_error_bar(axs[0, 2], x, labels,
                   y_gpu_icp, e_gpu_icp, y_cpu_icp, e_cpu_icp,
                   r'\textbf{Multi-scale Colored ICP}',
                   r'\textbf{Average time per submap pair} (ms)')

    draw_error_bar(axs[1, 0], x, labels,
                   y_gpu_fgr, e_gpu_fgr, y_cpu_fgr, e_cpu_fgr,
                   r'\textbf{Fast Global Registration}',
                   r'\textbf{Average time per submap pair} (ms)')
    draw_error_bar(axs[1, 1], x, labels,
                   y_gpu_fe, e_gpu_fe, y_cpu_fe, e_cpu_fe,
                   r'\textbf{FPFH Feature Extraction}',
                   r'\textbf{Average time per submap} (ms)')
    draw_error_bar(axs[1, 2], x, labels,
                   y_gpu_fm, e_gpu_fm, y_cpu_fm, e_cpu_fm,
                   r'\textbf{FPFH Feature Matching}',
                   r'\textbf{Average time per submap pair} (ms)')

    fig.tight_layout()

    # bar_cpu_odom = plt.bar(x + offset, y_cpu_odom, width=width)
    # plt.errorbar(x + offset, y_cpu_odom, yerr=e_cpu_odom, fmt='.g', capsize=20)
    # for i, v in enumerate(y_cpu_odom):
    #     plt.text(x[i] + offset + 0.02, y_cpu_odom[i] + 5, str(y_cpu_odom[i]))

    plt.savefig('time.pdf', bbox_inches='tight')
    plt.show()

