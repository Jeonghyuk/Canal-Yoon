#############################################################################
##
## Canal: Calcium imaging ANALyzer
##
## Copyright (C) 2015-2017 Youngtaek Yoon <caviargithub@gmail.com>
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
import scipy.ndimage.filters
import itertools
import skimage.feature
import skimage.morphology
import skimage.filters
import multiprocessing as mp
import canal.image.registration   # lattice_points
import tqdm
import canal.opt

def adaptive_otsu(image, order, n_proc=None):
    if n_proc is None:
        n_proc = mp.cpu_count() - 1

    # calculate otsu threshold around the points and store the value
    # in the corresponding points in the buffer.
    buf = np.zeros_like(image)
    # points = itertools.product(*[range(s) for s in image.shape])
    points = canal.registration.lattice_points(image.shape, order, order)

    if n_proc == 1:
        for point in points:
            index = tuple(slice(max(0, c - order), c + order + 1)
                          for c in point)
            buf[point] = skimage.filters.threshold_otsu(image[index])
    else:
        with mp.Pool(processes=n_proc) as pool:
            indices = (tuple(slice(max(0, c - order), c + order + 1)
                             for c in point)
                       for point in points)
            args = (image[index] for index in indices)
            for point, ret in zip(points,
                                  pool.imap(skimage.filters.threshold_otsu,
                                            args)):
                buf[point] = ret
    return buf

def ball(principal_axes):
    """
    Returns n-dimensional ball.
    """
    center = np.asarray(principal_axes) // 2
    radii = center + 0.5

    ranges = [range(elem) for elem in principal_axes]
    points = np.array(list(itertools.product(*ranges)))
    normed_diff = (points - center) / radii

    serial_kernel = np.sum(normed_diff * normed_diff, axis=-1) < 1
    return serial_kernel.reshape(principal_axes)

def _peak_local_max(args):
    image, threshold_abs, footprint = args
    return skimage.feature.peak_local_max(image, threshold_abs=threshold_abs,
                                          footprint=footprint)

def local_min(image, radius, partition, verbose=False):
    # create sub-images using partition
    splitter, setter, getter = canal.opt.split(image.shape, partition,
                                               (radius,) * image.ndim)
   
    # find peaks
    if verbose:
        print('Detecting blobs')
    footprint = ball((radius * 2 + 1,) * image.ndim)
    with mp.Pool(processes=np.product(partition)) as pool:
        args = ((-image[index], 0, footprint) for index in splitter)
        peak_groups = pool.map(_peak_local_max, args)

    # apply offset
    for anchor, loffset, group in zip(setter, getter, peak_groups):
        if len(group) != 0:
            offset = (np.array([i.start for i in anchor]) -
                      np.array([i.start for i in loffset]))
            group += offset
    centers = np.concatenate(peak_groups)
    return centers

def find_cells(image, radius, partition):
    """
    Finds cells in the image.

    Parameters
    ----------
    image: ndarray
    radius: float or tuple
        Radii of the cells. The condition ``len(radius) == image.ndim`` must
        be fulfilled. If float, it would be converted into a tuple.
    partition: tuple
        For parallel computing purpose. The condition ``len(partition) ==
        image.ndim`` must be fulfilled.

    Returns
    -------
    centers: list
    """
    if not isinstance(radius, (list, tuple)): # convert into a tuple
        radius = (radius,) * image.ndim

    # create sub-images using partition
    subinterval = tuple(s // p for s, p in zip(image.shape, partition))
    subindices = [tuple(slice(max(0, p * i - r), (p + 1) * i + r)
                        for p, i, r in zip(point, subinterval, radius))
                  for point in itertools.product(*[range(g) for g in partition])]

    # estimate background level
    bg_mean, bg_std = med_info(image, 3)
    bg_thres = bg_mean + bg_std * 3

    # find peaks
    with mp.Pool(processes=np.product(partition)) as pool:
        footprint = ball(np.asarray(radius) * 2 + 1)    # not typo
        args = ((image[index], bg_thres, footprint) for index in subindices)
        peak_groups = pool.map(_peak_local_max, args)

    # apply offset
    for subindex, group in zip(subindices, peak_groups):
        if len(group) != 0:
            offset = np.array([i.start for i in subindex])
            group += offset
    return np.concatenate(peak_groups)

def _peak_local_max_new(args):
    image, radius = args
    radiusnd = np.asarray(radius)

    lap = scipy.ndimage.filters.gaussian_laplace(image, radiusnd / 4)
    footprint = ball(radiusnd * 2 + 1)
    return skimage.feature.peak_local_max(-lap, threshold_abs=0,
                                          footprint=footprint)
def _gaussian_laplace(args):
    return scipy.ndimage.filters.gaussian_laplace(*args)

def scored(image, max_std):
    stack = np.empty((max_std - 1,) + image.shape)
    n_proc = min(mp.cpu_count() - 1, len(stack))
    with mp.Pool(processes=n_proc) as pool:
        args = ((image, std) for std in range(1, max_std))
        with tqdm.tqdm(args, total=len(stack)) as pbar:
            for num, result in enumerate(pool.imap(_gaussian_laplace,
                                                   args)):
                stack[num] = result
    return stack

def find_cells_new(image, std, partition, verbose=False):
    # laplacian of gaussian
    if verbose:
        print('Calculating laplacian of gaussian')
    lap = scipy.ndimage.filters.gaussian_laplace(image, std)

    radius = 0.5 * std * 2 ** 0.5
    # create sub-images using partition
    splitter, setter, getter = canal.opt.split(lap.shape, partition,
                                               (int(radius),) * image.ndim)
   
    # find peaks
    if verbose:
        print('Detecting blobs')
    footprint = ball((int(radius) * 2 + 1,) * image.ndim)
    with mp.Pool(processes=np.product(partition)) as pool:
        args = ((-lap[index], 0, footprint) for index in splitter)
        peak_groups = pool.map(_peak_local_max, args)

    # apply offset
    for anchor, loffset, group in zip(setter, getter, peak_groups):
        if len(group) != 0:
            offset = (np.array([i.start for i in anchor]) -
                      np.array([i.start for i in loffset]))
            group += offset
    centers = np.concatenate(peak_groups)

    # watershed
    random_centers = centers.copy()
    np.random.shuffle(random_centers)
    markers = np.zeros(image.shape, int)
    for index, center in zip(itertools.count(1), random_centers):
        markers[tuple(center)] = index
    ws_label = skimage.morphology.watershed(lap, markers)
    return centers, ws_label

def dev(data, center):
    """
    Returns deviation
    """
    diff = data - center
    var = np.mean(diff * diff)
    return np.sqrt(var)

def med_info(image, sigma):
    """
    Get information around median.

    Parameters
    ----------
    image: ndarray
    sigma: float

    Returns
    -------
    mean: float
    std: float
    """
    image_med = int(np.median(image))
    hist_width = int(dev(image, image_med) * sigma)
    hist_edges = np.linspace(image_med - hist_width - 0.5,
                             image_med + hist_width + 0.5,
                             hist_width * 2 + 2)

    hist = np.histogram(image, hist_edges)[0]
    edge_centers = 0.5 * (hist_edges[:-1] + hist_edges[1:])
    dense_at = edge_centers[hist.argmax()]
    diff = edge_centers - edge_centers[hist.argmax()]
    var = np.sum(hist * diff * diff) / hist.sum()
    return dense_at, np.sqrt(var)
    
def find_cells_legacy(image, radius):
    # get background
    image_min, image_mean = int(image.min()), int(image.mean())
    hist_edges = np.linspace(image_min - 0.5, image_mean + 0.5,
                             image_mean - image_min + 2)
    hist = np.histogram(image, hist_edges)[0]
    
    def find_reverse(vec):
        if vec[0] < 0:
            vec = -vec
        for arg, elem in enumerate(vec):
            if elem < 0:
                return arg
        else:
            raise Exception('No reverse point.')
    
    arg_left_half = find_reverse(hist - hist.max() * 0.5)
    arg_max = hist.argmax()
    half_width = arg_max - arg_left_half
    
    hist_middle = (hist_edges[1:] + hist_edges[:-1]) * 0.5
    background_max = hist_middle[arg_max + half_width]
    background_mask = image < background_max
    
    gauss_filtered = scipy.ndimage.filters.gaussian_filter(image, radius)
    laplacian_filtered = scipy.ndimage.filters.laplace(-gauss_filtered)
    
    footprint = ball((radius * 2 + 1,) * image.ndim)
    background_lap_std = laplacian_filtered[background_mask].std()
    threshold = background_lap_std * 4  # 0.000063
    positions = skimage.feature.peak_local_max(laplacian_filtered, radius * 2 + 1,
                                               threshold, 0, footprint = footprint)
    
    def indices_to_mask(indices, shape):
        mask = np.zeros(shape, bool)
        for index in indices:
            mask[tuple(index)] = True
        return mask
    
    #label_mask = indices_to_mask(positions, image.shape)
    #basin_markers, n_markers = scipy.ndimage.label(label_mask)
    #labels = skimage.morphology.watershed(-laplacian_filtered, basin_markers,
    #                                      mask = ~background_mask)
    #return positions, labels
    return positions
    
def show_result(image, points, radius):
    ret = image.copy()
    mark_val = image.max()
    for point in points:
        z, y, x = point
        ret[z - radius:z + radius + 1,
            y - radius:y + radius + 1,
            x - radius:x + radius + 1] = mark_val
    
    return ret

def _shrink(args):
    frame, indices, masks = args
    result = np.empty(len(masks))
    for num, (index, mask) in enumerate(zip(indices, masks)):
        result[num] = np.sum(frame[index] * mask) / mask.sum()

def extract(movie, centers, radius, transform, n_proc=None, verbose=False):
    """
    Extracts the time series of intensity of ROIs (region of interests)
    reconstructed using the centers and the radius from the movie.

    Parameters
    ----------
    movie: ndarray
        ``movie.ndim`` == D and ``len(movie)`` == T.
    centers: array-like of shape (N, D - 1)
        The centers of the nuclei (the number of nuclei is N).
    radius: float
    transform: Translation
    n_proc: int, optional
    verbose: bool, optional

    Returns
    -------
    signals: ndarray of shape (N, T)
    """
    if n_proc is None:
        n_proc = mp.cpu_count() - 1

    diameter = radius * 2 + 1
    n_times, n_heights = movie.shape[:2]
    # get masks
    if verbose:
        print('Creating masks')
    with tqdm.tqdm(centers, unit='cell', disable=not verbose) as pbar:
        indices, masks = [], []
        valid_centers = []
        for cell, center in enumerate(pbar):
            begin = np.asarray(center) - radius
            end = np.asarray(center) + radius
            inverse = transform.inverse()
            begin_sparse = inverse(begin)
            end_sparse = inverse(end)
            begin_yx_sparse = np.ceil(begin_sparse[1:]).astype(int)
            end_yx_sparse = (np.floor(end_sparse[1:]) + 1).astype(int)
            begin_z_sparse = int(np.ceil(begin_sparse[0]))
            end_z_sparse = int(np.floor(end_sparse[0])) + 1
            
            index = (slice(begin_z_sparse, end_z_sparse),
                     slice(begin_yx_sparse[0], end_yx_sparse[0]),
                     slice(begin_yx_sparse[1], end_yx_sparse[1]))

            # ignore cells outside of the movie
            is_outside = any(i.start < 0 for i in index) or \
                         any(i.stop > s for i, s in zip(index, movie.shape[1:]))
            if is_outside:
                continue
            else:
                valid_centers.append(center)
            mask = np.empty((end_z_sparse - begin_z_sparse, diameter, diameter),
                            movie.dtype)
            
            morp = ball((diameter,) * 2)
            for num, z_sparse in enumerate(range(begin_z_sparse, end_z_sparse)):
                z = int(round(transform((z_sparse, 0, 0))[0]))
                mask[num] = morp[z - center[0]]
            indices.append(index)
            masks.append(mask)

    # get means
    n_cells = len(valid_centers)
    signals = np.empty((n_cells, n_times))
    if verbose:
        print('Extracting signals: {} cells'.format(n_cells))
    with tqdm.tqdm(movie, unit='frame', disable=not verbose) as pbar:
        for time, frame in enumerate(pbar):
            for cell, (index, mask) in enumerate(zip(indices, masks)):
                signals[cell, time] = np.sum(frame[index] * mask) / mask.sum()


    return valid_centers, signals
