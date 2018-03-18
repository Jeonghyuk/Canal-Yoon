import numpy as _np
from scipy.ndimage.filters import gaussian_filter1d as _gfilter
from sklearn.decomposition import PCA as _pca
from mpl_toolkits.mplot3d import Axes3D as _3ax
import matplotlib.pyplot as _plt

def pca(mat, n_components):
    pca_solver = _pca(n_components = n_components).fit(mat)
    return (pca_solver.transform(mat), pca_solver.components_)

def linear_color_sequence(begin, end, n_points):
    mat = _np.empty((n_points, 3))
    for component in range(3):
        mat[:, component] = _np.linspace(begin[component], end[component], num = n_points, endpoint = False)
    return mat

def spliced_linear_color_sequence(colors, n_points):
    return _np.concatenate([linear_color_sequence(colors[num], colors[num + 1], n_points[num]) for num in range(len(n_points))], axis = 0)
