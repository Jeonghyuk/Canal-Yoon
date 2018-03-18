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
import skimage.morphology
import skimage.filters
import canal.cell
import itertools
import tqdm
import scipy.stats
import scipy.interpolate
import scipy.ndimage
import skimage.util
import math
import multiprocessing as mp

def blob_log(image, min_sigma=1, max_sigma=50, num_sigma=10, threshold=0,
             mask=None, overlap=.5, ratio=np.sqrt(2), log_scale=False):
    """
    Finds blobs in the given grayscale image.

    Blobs are found using the Laplacian of Gaussian (LoG) method [1]_.
    For each blob found, the method returns its coordinates and the standard
    deviation of the Gaussian kernel that detected the blob.

    Parameters
    ----------
    image : ndarray
        Input grayscale image, blobs are assumed to be light on dark
        background (white on black).
    min_sigma : float, optional
        The minimum standard deviation for Gaussian Kernel. Keep this low to
        detect smaller blobs.
    max_sigma : float, optional
        The maximum standard deviation for Gaussian Kernel. Keep this high to
        detect larger blobs.
    num_sigma : int, optional
        The number of intermediate values of standard deviations to consider
        between `min_sigma` and `max_sigma`.
    threshold : float, optional.
        The absolute lower bound for scale space maxima. Local maxima smaller
        than thresh are ignored. Reduce this to detect blobs with less
        intensities.
    overlap : float, optional
        A value between 0 and 1. If the area of two blobs overlaps by a
        fraction greater than `threshold`, the smaller blob is eliminated.
    log_scale : bool, optional
        If set intermediate values of standard deviations are interpolated
        using a logarithmic scale to the base `10`. If not, linear
        interpolation is used.

    Returns
    -------
    A : (n, 3) ndarray
        A 2d array with each row representing 3 values, ``(y,x,sigma)``
        where ``(y,x)`` are coordinates of the blob and ``sigma`` is the
        standard deviation of the Gaussian kernel which detected the blob.

    References
    ----------
    .. [1] http://en.wikipedia.org/wiki/Blob_detection#The_Laplacian_of_Gaussian

    Examples
    --------
    >>> from skimage import data, feature, exposure
    >>> img = data.coins()
    >>> img = exposure.equalize_hist(img)  # improves detection
    >>> feature.blob_log(img, threshold = .3)
    array([[ 113.        ,  323.        ,    1.        ],
           [ 121.        ,  272.        ,   17.33333333],
           [ 124.        ,  336.        ,   11.88888889],
           [ 126.        ,   46.        ,   11.88888889],
           [ 126.        ,  208.        ,   11.88888889],
           [ 127.        ,  102.        ,   11.88888889],
           [ 128.        ,  154.        ,   11.88888889],
           [ 185.        ,  344.        ,   17.33333333],
           [ 194.        ,  213.        ,   17.33333333],
           [ 194.        ,  276.        ,   17.33333333],
           [ 197.        ,   44.        ,   11.88888889],
           [ 198.        ,  103.        ,   11.88888889],
           [ 198.        ,  155.        ,   11.88888889],
           [ 260.        ,  174.        ,   17.33333333],
           [ 263.        ,  244.        ,   17.33333333],
           [ 263.        ,  302.        ,   17.33333333],
           [ 266.        ,  115.        ,   11.88888889]])
    Notes
    -----
    The radius of each blob is approximately :math:`\sqrt{2}sigma`.
    """
    if log_scale:
        start, stop = math.log(min_sigma, 10), math.log(max_sigma, 10)
        sigma_list = np.logspace(start, stop, num_sigma)
    else:
        sigma_list = np.linspace(min_sigma, max_sigma, num_sigma)

    # computing gaussian laplace
    # s provides scale invariance
    image_cube = np.empty((len(sigma_list),) + image.shape, image.dtype)
    with mp.Pool(processes=min(len(sigma_list), mp.cpu_count())) as pool:
        args = ((image, s) for s in sigma_list)
        with tqdm.tqdm(args, total=len(sigma_list), unit='sigma') as pbar:
            for index, res in enumerate(pool.imap(_invariant_laplace, pbar)):
                image_cube[index] = res

    footprint = np.ones((3,) * image_cube.ndim, bool)
    local_maxima = skimage.feature.peak_local_max(image_cube,
                                                  threshold_abs=threshold,
                                                  footprint=footprint,
                                                  threshold_rel=0.0,
                                                  exclude_border=True)

    if mask is not None:
        local_maxima = np.array([elem for elem in local_maxima
                                 if mask[tuple(elem[1:])]])

    # Catch no peaks
    if len(local_maxima) == 0:
        return np.empty((0,3))
    else:
        print('{} local maxima detected.'.format(len(local_maxima)))

    # Calculate the radii
    radii = sigma_list[local_maxima[:, 0]]

    # Convert the last index to its corresponding scale value
    local_maxima = np.c_[local_maxima[:, 1:], radii]
    return _prune_blobs(local_maxima, overlap, ratio)

def _invariant_laplace(args):
    image, sigma = args
    return -sigma ** 2 * scipy.ndimage.filters.gaussian_laplace(image, sigma)

def _prune_kernel(args):
    blobs_array, overlap, ratio, begin, end = args
    for index1 in range(begin, end):
        blob1 = blobs_array[index1]
        for index2 in range(index1 + 1, len(blobs_array)):
            blob2 = blobs_array[index2]
            if _blob_overlap(blob1, blob2, ratio) > overlap:
                if blob1[-1] > blob2[-1]:
                    blob2[-1] = -1
                else:
                    blob1[-1] = -1
    return blobs_array[begin:end]

def _prune_blobs(blobs_array, overlap, ratio):
    """
    Eliminated blobs with area overlap.

    Parameters
    ----------
    blobs_array : ndarray
        A 2d array with each row representing 3 values, ``(y,x,sigma)``
        where ``(y,x)`` are coordinates of the blob and ``sigma`` is the
        standard deviation of the Gaussian kernel which detected the blob.
    overlap : float
        A value between 0 and 1. If the fraction of area overlapping for 2
        blobs is greater than `overlap` the smaller blob is eliminated.

    Returns
    -------
    A : ndarray
        `array` with overlapping blobs removed.
    """
    '''
    # iterating again might eliminate more blobs, but one iteration suffices
    # for most cases
    for blob1, blob2 in itertools.combinations(blobs_array, 2):
        if _blob_overlap(blob1, blob2) > overlap:
            if blob1[-1] > blob2[-1]:
                blob2[-1] = -1
            else:
                blob1[-1] = -1
    return np.array([b for b in blobs_array if b[-1] > 0])
    '''
    n_proc = 32
    n_blobs = len(blobs_array)
    work_borders = np.r_[[0], 
                         n_blobs * np.sqrt(np.arange(1, n_proc) / n_proc),
                         [n_blobs]]
    work_borders = n_blobs - work_borders.astype(int)[::-1]
    with mp.Pool(processes=n_proc) as pool:
        args = ((blobs_array, overlap, ratio, begin, end)
                for begin, end in zip(work_borders[:-1], work_borders[1:]))
        res = pool.map(_prune_kernel, args)

    # return blobs_array[blobs_array[:, 2] > 0]
    return np.array([b for b in np.concatenate(res) if b[-1] > 0])

def _blob_overlap(blob1, blob2, ratio):
    """
    Finds the overlapping area fraction between two blobs.

    Returns a float representing fraction of overlapped area.

    Parameters
    ----------
    blob1 : sequence
        A sequence of ``(z,y,x,sigma)``, where ``x,y,z`` are coordinates of blob
        and sigma is the standard deviation of the Gaussian kernel which
        detected the blob.
    blob2 : sequence
        A sequence of ``(z,y,x,sigma)``, where ``x,y,z`` are coordinates of blob
        and sigma is the standard deviation of the Gaussian kernel which
        detected the blob.

    Returns
    -------
    f : float
        Fraction of overlapped area.
    """
    root2 = ratio

    # extent of the blob is given by sqrt(2)*scale
    r1 = blob1[-1] * root2
    r2 = blob2[-1] * root2

    d = np.linalg.norm(blob2[:-1] - blob1[:-1])

    if d > r1 + r2:
        return 0

    # one blob is inside the other, the smaller blob must die
    if d <= abs(r1 - r2):
        return 1

    cos1 = (d ** 2 + r1 ** 2 - r2 ** 2) / (2 * d * r1)
    cos2 = (d ** 2 + r2 ** 2 - r1 ** 2) / (2 * d * r2)

    volume1 = r1 ** 3 * (2 - cos1 * (1 + cos1 ** 2))
    volume2 = r2 ** 3 * (2 - cos2 * (1 + cos2 ** 2))

    return (volume1 + volume2) / (4 * min(r1, r2) ** 3)

def threshold_background2(image, edges, verbose=False):
    scores = []
    with tqdm.tqdm(zip(edges[:-1], edges[1:]),
                   total=len(edges) - 1, disable=not verbose) as pbar:
        for num, (l, h) in enumerate(pbar):
            mask = (l <= image) & (image < h)
            dil = skimage.morphology.binary_dilation(mask)
            scores.append(dil.sum())
    return scores

def threshold_background(image, edges, verbose=False, partition=(2, 4, 4)):
    scores = []
    with tqdm.tqdm(zip(edges[:-1], edges[1:]),
                   total=len(edges) - 1, disable=not verbose) as pbar:
        for num, (l, h) in enumerate(pbar):
            mask = (l <= image) & (image < h)
            dil = skimage.morphology.binary_dilation(mask)
            if num != 0:
                ratio = np.sum(prev_dil & mask) / prev_dil.sum()
                ratio_prev = np.sum(dil & prev_mask) / dil.sum()
                scores.append(ratio * ratio_prev)
            prev_mask, prev_dil = mask, dil
    if verbose:
        print('Matching: {:.2f}'.format(np.sqrt(max(scores))))
    return edges[1:-1][np.argmax(scores)]

def statistic_background(image, edges):
    # determine edges of the histogram
    #if isinstance(edges, int):
    #    edges = np.linspace(image.min(), image.mean(), edges)

    hist, edges = np.histogram(image, edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    spline = scipy.interpolate.UnivariateSpline(centers,
                                                hist - hist.max() * 0.5)

    # narrow down to baseline level
    roots = spline.roots()  # could be more than 2 of them because of the noise
    peak = centers[hist.argmax()]
    left = roots[roots < peak][-1]
    right = roots[roots > peak][0]
    narrowed = image[((image < right) & (image > left))]

    # gaussian approximation
    return scipy.stats.norm.fit(narrowed)    # mean, std

def foreground2d(soma, scale):
    global_bright = soma > np.median(soma)
    local_bright = soma > skimage.filters.gaussian(soma, scale)
    ball = canal.cell.ball((scale * 2 + 1,) * soma.ndim)
    opening = skimage.morphology.binary_opening(global_bright & local_bright,
                                                ball)
    return opening

def featurescore(soma, scale):
    blur = skimage.filters.gaussian(soma, scale)
    sobel = scipy.ndimage.generic_gradient_magnitude(blur, scipy.ndimage.sobel)
    return sobel
    ball = canal.cell.ball((scale * 2 + 1,) * soma.ndim)
    opening = skimage.morphology.binary_opening(global_bright & local_bright,
                                                ball)
    return opening

def foreground3d(soma, nuclear, scale, verbose=False):
    def dev(data, center):
        diff = data - center
        return np.sqrt(np.mean(diff * diff))

    def outside(data, center, width):
        return (data > center + width) | (data < center - width)

    ## background mask
    if verbose:
        print('Creating background mask')
    soma = soma.astype(float)
    nuclear = nuclear.astype(float)
    #ratio = soma / nuclear
    '''
    # distribution of background intensities
    hist, edges = np.histogram(ratio, np.linspace(ratio.min(),
                                                  ratio.mean(), 512))
    centers = 0.5 * (edges[:-1] + edges[1:])
    bg_spline = scipy.interpolate.UnivariateSpline(centers,
                                                   hist - hist.max() / 2)
    bg_left, bg_right = bg_spline.roots()   # narrow
    bg_vals = ratio[((ratio < bg_right) & (ratio > bg_left))]

    # gaussian approximation
    bg_center, bg_std = scipy.stats.norm.fit(bg_vals)
    bg_mask = outside(ratio, bg_center, bg_std * 3) # bg zero
    '''
    # rough border mask
    if verbose:
        print('Creating border mask')
    footprint = canal.cell.ball((scale * 2 + 1,) * nuclear.ndim)
    nuclearblur = skimage.filters.gaussian(nuclear, scale)
    somablur = skimage.filters.gaussian(soma, scale)
    nuclearmask_rough = nuclear > nuclearblur
    #somamask_rough = soma > somablur
    #bordermask_rough = somamask_rough & nuclearmask_rough
    #bordermask = skimage.morphology.binary_opening(bordermask_rough, footprint)

    # rough mask for whole cns
    #cnsmask_rough = somablur > somablur[bordermask].mean()

    somasobel = scipy.ndimage.generic_gradient_magnitude(somablur,
                                                         scipy.ndimage.sobel)
    hist, edges = np.histogram(somablur, 512, weights=somasobel ** 2)
    centers = 0.5 * (edges[:-1] + edges[1:])
    cnsmask = somablur > centers[hist.argmax()]

    nuclearmask = skimage.morphology.binary_closing(nuclearmask_rough & cnsmask,
                                                    footprint)
    neuropilmask = skimage.morphology.binary_opening(cnsmask & ~nuclearmask,
                                                     footprint)
    return cnsmask, neuropilmask
