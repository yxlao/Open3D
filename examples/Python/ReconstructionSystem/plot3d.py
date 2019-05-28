import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D


def ax3d(fig=None):
    if fig is None:
        fig = plt.gcf()
    return fig.add_subplot(111, projection="3d")


def cameracenter_from_translation(R, t):
    # - R.T @ t
    t = t.reshape(-1, 3, 1)
    R = R.reshape(-1, 3, 3)
    C = -R.transpose(0, 2, 1) @ t
    return C.squeeze()


def cameracenter_from_T(T):
    R, t = T[:3, :3], T[:3, 3]
    return cameracenter_from_translation(R, t)


def axis_equal(ax=None):
    if ax is None:
        ax = plt.gca()
    extents = np.array(
        [getattr(ax, "get_{}lim".format(dim))() for dim in "xyz"])
    sz = extents[:, 1] - extents[:, 0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize / 2
    for ctr, dim in zip(centers, "xyz"):
        getattr(ax, "set_{}lim".format(dim))(ctr - r, ctr + r)


def axis_label(ax=None, x="x", y="y", z="z"):
    if ax is None:
        ax = plt.gca()
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_zlabel(z)


def plot_camera(ax=None,
                R=np.eye(3),
                t=np.zeros((3,)),
                size=25,
                marker_C=".",
                color="b",
                linestyle="-",
                linewidth=0.1,
                label=None,
                txt=None,
                **kwargs):
    if ax is None:
        ax = plt.gca()
    C0 = cameracenter_from_translation(R, t).ravel()
    C1 = (C0 + R.T.dot(
        np.array([[-size], [-size], [3 * size]], dtype=np.float32)).ravel())
    C2 = (C0 + R.T.dot(
        np.array([[-size], [+size], [3 * size]], dtype=np.float32)).ravel())
    C3 = (C0 + R.T.dot(
        np.array([[+size], [+size], [3 * size]], dtype=np.float32)).ravel())
    C4 = (C0 + R.T.dot(
        np.array([[+size], [-size], [3 * size]], dtype=np.float32)).ravel())

    if marker_C != "":
        ax.plot([C0[0]], [C0[1]], [C0[2]],
                marker=marker_C,
                color=color,
                label=label,
                **kwargs)
    ax.plot([C0[0], C1[0]], [C0[1], C1[1]], [C0[2], C1[2]],
            color=color,
            label="_nolegend_",
            linestyle=linestyle,
            linewidth=linewidth,
            **kwargs)
    ax.plot([C0[0], C2[0]], [C0[1], C2[1]], [C0[2], C2[2]],
            color=color,
            label="_nolegend_",
            linestyle=linestyle,
            linewidth=linewidth,
            **kwargs)
    ax.plot([C0[0], C3[0]], [C0[1], C3[1]], [C0[2], C3[2]],
            color=color,
            label="_nolegend_",
            linestyle=linestyle,
            linewidth=linewidth,
            **kwargs)
    ax.plot([C0[0], C4[0]], [C0[1], C4[1]], [C0[2], C4[2]],
            color=color,
            label="_nolegend_",
            linestyle=linestyle,
            linewidth=linewidth,
            **kwargs)
    ax.plot([C1[0], C2[0], C3[0], C4[0], C1[0]],
            [C1[1], C2[1], C3[1], C4[1], C1[1]],
            [C1[2], C2[2], C3[2], C4[2], C1[2]],
            color=color,
            label="_nolegend_",
            linestyle=linestyle,
            linewidth=linewidth,
            **kwargs)

    if txt is not None:
        ax.text(*C0, txt)


def plot_cameras(Ts, size=25):
    plt.figure()
    ax3d()
    for idx, T in enumerate(Ts):
        R, t = T[:3, :3], T[:3, 3]
        plot_camera(R=R, t=t, size=size, txt=f"{idx:02d}", linewidth=1.0)
    axis_equal()
    axis_label()
    plt.show()
