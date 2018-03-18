#############################################################################
##
## Canal: Calcium imaging ANALyzer
##
## Copyright (C) 2015-2016 Youngtaek Yoon <caviargithub@gmail.com>
##
## This file is part of the source code of Canal.
##
## This program is free software: you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
##
## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with this program.  If not, see <http://www.gnu.org/licenses/>.
##
#############################################################################

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import mpl_toolkits.mplot3d as mp3d
import itertools

def imshow_tile(times, data, out=False, **kwargs):
    times = np.asarray(times)
    if times.ndim == 1:
        times = times.reshape(1, times.size)
    tile = np.empty(tuple(ts * ds for ts, ds
                          in zip(times.shape, data.shape[1:])), data.dtype)
    for pos in itertools.product(range(times.shape[0]), range(times.shape[1])):
        index = tuple(slice(p * s, (p + 1) * s)
                      for p, s in zip(pos, data.shape[1:]))
        tile[index] = data[times[pos]]

    if out:
        return tile
    plt.imshow(tile, **kwargs)

def preset_colors(name):
    """
    Returns preset color table.

    Parameters
    ----------
    name: string
        The name of table. Available tables are:
        colorblind10: 
    """
    if name == 'colorblind10':
        return np.array([[255, 128,  14],
                         [171, 171, 171],
                         [ 95, 158, 209],
                         [ 89,  89,  89],
                         [  0, 107, 164],
                         [255, 188, 121],
                         [207, 207, 207],
                         [200,  82,   0],
                         [162, 200, 236],
                         [137, 137, 137]]) / 255

def plots(arr, space, repel=True, **kwargs):
    if repel:
        plt.plot(arr[0], **kwargs)
        before = arr[0]
        for cur in arr[1:]:
            contact_at = (cur - before).argmin()
            before = cur - cur[contact_at] + before[contact_at] + space
            plt.plot(before, **kwargs)
        plt.show()
    else:
        for num, cur in enumerate(arr):
            plt.plot(cur + space * num, **kwargs)
        plt.show()

def fill_spheres(blobs, shape):
    import canal.cell
    ret = np.zeros(shape, int)
    for index, blob in enumerate(blobs):
        center, radius = blob[:-1].astype(int), int(blob[-1])
        ball = canal.cell.ball((2 * radius + 1,) * ret.ndim)
        begin = tuple(c - radius for c in center)
        end = tuple(c + radius + 1 for c in center)
        mask = tuple(slice(b, e) for b, e in zip(begin, end))
        try:
            ret[mask][ball] = index + 1
        except:
            print(center)
    return ret

def scatter3d(mat, append=False, **kwargs):
    z, y, x = mat.T
    fig = plt.figure()
    ax = mp3d.Axes3D(fig)
    ax.scatter(x, y, z, **kwargs)
    
    # get lim
    cube_begin = np.array([x.min(), y.min(), z.min()])
    cube_end = np.array([x.max(), y.max(), z.max()])
    arm = np.max(cube_end - cube_begin) * 0.5
    x_mid, y_mid, z_mid = 0.5 * (cube_begin + cube_end)

    ax.set_xlim(x_mid - arm, x_mid + arm)
    ax.set_ylim(y_mid - arm, y_mid + arm)
    ax.set_zlim(z_mid - arm, z_mid + arm)
    plt.show()

def plot3d(mat, **kwargs):
    z, y, x = mat.T
    fig = plt.figure()
    ax = mp3d.Axes3D(fig)
    ax.plot(x, y, z, **kwargs)
    plt.show()

def orthogonal_view(image, point, **kwargs):
    z, y, x = np.asarray(point).round().astype(int)
    zend, yend, xend = image.shape
    gs = plt.GridSpec(2, 2, hspace=0, wspace=0, width_ratios=(xend, zend),
                      height_ratios=(yend, zend))

    # xy-plane
    ax0 = plt.subplot(gs[0])
    ax0.plot([x] * 2, [0, yend], 'y')
    ax0.plot([0, xend], [y] * 2, 'y')
    ax0.imshow(image[z], **kwargs)
    ax0.xaxis.tick_top()
    ax0.xaxis.set_label_position('top')
    ax0.set_xlabel('X')
    ax0.set_ylabel('Y')

    # yz-plane
    ax1 = plt.subplot(gs[1])#, sharey=ax0)
    ax1.plot([z] * 2, [0, yend], 'y')
    ax1.plot([0, zend], [y] * 2, 'y')
    ax1.imshow(image[..., x].T, **kwargs)
    ax1.yaxis.tick_right()
    ax1.yaxis.set_label_position('right')
    ax1.set_xlabel('Z')
    ax1.set_ylabel('Y')

    # xz-plane
    ax2 = plt.subplot(gs[2])#, sharex=ax0)
    ax2.plot([x] * 2, [0, zend], 'y')
    ax2.plot([0, xend], [z] * 2, 'y')
    ax2.imshow(image[:, y], **kwargs)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Z')

    plt.show()

def colorplot3d(mat, colors=None, ctable=None, **kwargs):
    """
    Plot 3d data colored.

    Parameters
    ----------
    mat: ndarray of shape (3, N)
        Plotting data. mat[0], mat[1] and mat[2] would be interpreted as
        x-coordinates, y-coordinates and z-coordinates respectively.
    colors: int ndarray of shape(N,) or ndarray of shape (N, 3)
        Color indices or list of RGB colors. If colors is None,
        black would be used.
    ctable: ndarray of shape (L, 3)
        Color table used converting color index to RGB color. If ctable is None,
        colors would be interpreted as list of RGB colors.

    Examples
    --------
    >>> ctable = np.array([[255, 0, 0],
                           [0, 255, 0],
                           [0, 0, 255]])    # index 0, 1 and 2 converted to
                                            # red, green and blue respectively
    >>> cindices = [0, 0, 0, 1, 2]  # red, red, red, green, blue
    >>> data = np.arange(15).reshape((3, 5))
    >>> colorplot3d(data, cindices, ctable)

    """
    n_points = mat.shape[-1]
    if ctable is None:
        if colors is None:
            colors = ['k'] * n_points
        else:
            colors = np.asarray(colors)
    else:
        if ctable.dtype == int:
            ctable = ctable / 255
        color_indices = list(colors)
        colors = ctable[color_indices]

    plot3d(mat, c=colors, **kwargs)

def colorplot(xdata, ydata=None, cprofile=None, ctable=None, **kwargs):
    """
    Plot 2d data colored.

    Parameters
    ----------
    mat: ndarray of shape (3, N)
        Plotting data. mat[0], mat[1] and mat[2] would be interpreted as
        x-coordinates, y-coordinates and z-coordinates respectively.
    colors: int ndarray of shape(N,) or ndarray of shape (N, 3)
        Color indices or list of RGB colors. If colors is None,
        black would be used.
    ctable: ndarray of shape (L, 3)
        Color table used converting color index to RGB color. If ctable is None,
        colors would be interpreted as list of RGB colors.

    Examples
    --------
    >>> ctable = np.array([[255, 0, 0],
                           [0, 255, 0],
                           [0, 0, 255]])    # index 0, 1 and 2 converted to
                                            # red, green and blue respectively
    >>> cindices = [0, 0, 0, 1, 2]  # red, red, red, green, blue
    >>> data = np.arange(15).reshape((3, 5))
    >>> colorplot3d(data, cindices, ctable)

    """
    # if ydata only
    if ydata is None:
        ydata = xdata
        xdata = range(len(ydata))

    n_lines = len(ydata) - 1
    if ctable is None:
        if cprofile is None:
            colors = ['k'] * n_points
        else:
            colors = np.asarray(cprofile)
    else:
        if ctable.dtype == int:
            ctable = ctable / 255
        if cprofile.ndim == 1:
            color_indices = list(cprofile)
            colors = ctable[color_indices]
        elif cprofile.ndim == 2:
            trans = list(zip(ydata[:-1], ydata[1:]))
            color_indices = [cprofile[elem] for elem in trans]
            colors = ctable[color_indices]
        elif cprofile.ndim == 3:
            trans = list(zip(ydata[:-1], ydata[1:]))
            colors = [cprofile[elem] for elem in trans]

    for xl, yl, color in zip(zip(xdata[:-1], xdata[1:]),
                             zip(ydata[:-1], ydata[1:]), colors):
        plt.plot(xl, yl, c=color, **kwargs)

def colorbar(colors, top, bottom, xs=None, ctable=None):
    """
    Plot 2d data colored.

    Parameters
    ----------
    colors: int ndarray of shape(N,) or ndarray of shape (N, 3)
        Color indices or list of RGB colors.
    top: float
    bottom: float
    xs: ndarray of shape (N,)
        x-coordinates of the centers of the bars. If None (by default),
        would be ``np.arange(len(colors))``.
    ctable: ndarray of shape (L, 3)
        Color table used converting color index to RGB color. If ctable is None,
        colors would be interpreted as list of RGB colors.

    Examples
    --------
    >>> ctable = np.array([[255, 0, 0],
                           [0, 255, 0],
                           [0, 0, 255]])    # index 0, 1 and 2 converted to
                                            # red, green and blue respectively
    >>> cindices = [0, 0, 0, 1, 2]  # red, red, red, green, blue
    >>> colorbar(cindices, 0.0, 1.0, ctable)

    """
    n_points = len(colors)
    if ctable is None:
        if np.iterable(colors):
            if not isinstance(colors, np.ndarray):
                colors = np.array(colors)
            if colors.shape != (n_points, 3):
                raise ValueError('Invalid color format')
    else:
        if ctable.ndim != 2 or ctable.shape[-1] != 3:
            raise ValueError('Invalid color table format')
        else:
            if ctable.dtype == int:
                ctable = ctable / 255
        if not isinstance(colors, np.ndarray):
            colors = np.array(colors)
        if colors.shape != (n_points,) or colors.dtype != int:
            raise ValueError('Invalid color format')
        colors = ctable[colors]

    xs = np.arange(len(colors)) if xs is None else np.asarray(xs).reshape(xs.size)
    centers = 0.5 * (xs[1:] + xs[:-1])
    borders = np.concatenate((xs[:1], centers, xs[-1:]))
    for color, left, right in zip(colors, borders[:-1], borders[1:]):
        plt.bar(left, top - bottom, right - left, bottom,
                color=color, edgecolor=color)
