import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import matplotlib.collections as mcoll


def register_custom_colormap():
    cmap = LinearSegmentedColormap.from_list('ColdDarkHot',
                                             [(0, 1, 1), (0, 1, 1), (0, 1, 1), (0, 0, 1), (0, 0, 1), (0, 0, 1),
                                              (0, 0, 0), (1, 0, 0), (1, 0, 0), (1, 0, 0), (1, 1, 0), (1, 1, 0),
                                              (1, 1, 0)], N=256)
    plt.register_cmap(cmap=cmap)

register_custom_colormap()


def make_segments(x, y):
    a = np.array([x, y])
    xy_tuples = a.T  # generates tuples of [x1, y1] .. [xn, yn]
    packed_tuples = xy_tuples.reshape(-1, 1, 2)  # packs each tuple in a list : [x1, y1] -> [[x1, y1]]
    all_but_last = packed_tuples[:-1]  # all tuples but last
    all_but_first = packed_tuples[1:]  # all tuples but first
    segments = np.concatenate([all_but_last, all_but_first], axis=1)  # generates segments [[x1, y1], [x2, y2]] ... [xn-1, yn-1], [xn, yn]]

    return segments


def plot_multicolored_line(y, ax, z=None, x=None, linewidth=0.5, cmap='ColdDarkHot', default_color=(0, 0, 0), norm=None, black_and_white=False, thickness=False, adjustplot=True, adjustax=False):
    if x is None:
        x = range(len(y))

    if z is None:
        z = np.ones((len(x),))
        cmap = LinearSegmentedColormap.from_list('Mono', [default_color, default_color])

    if black_and_white:
        cmap = LinearSegmentedColormap.from_list('Mono', [default_color, default_color])
        thickness = True

    segments = make_segments(x, y)
    z = np.asarray(z)
    shifted_z = np.concatenate([z[1:], [z[-1]]])  # shift z one point to the left, duplicate last point to colorize lines based on their target values, not start values.

    if thickness:
        linewidths = shifted_z + 0.75
        # linewidths = shifted_z * shifted_z * shifted_z * 1.5 + 0.05  # high=thick
        # linewidths = (np.ones(np.shape(z)) * 1.75) - (z * z * z * 1.5)  # low=thick
    else:
        linewidths = (linewidth,)

    lc = mcoll.LineCollection(segments,
                              array=shifted_z,
                              cmap=cmap,
                              norm=norm,
                              linewidths=linewidths,
                              capstyle='round')

    ax.add_collection(lc)

    if adjustplot:
        plt.ylim(min(y), max(y)+0.00001)
        plt.xlim(min(x), max(x)+0.00001)

    if adjustax:
        ax.set_ylim(min(y), max(y))
        ax.set_xlim(min(x), max(x))

    return lc
