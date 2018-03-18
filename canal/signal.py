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
import canal.local
import multiprocessing as mp
import tqdm
import scipy.stats
import scipy.interpolate
import canal.image.esti

def _sharpen_kernel(vec, width):
    label, label_max = scipy.ndimage.label(vec)
    ret = np.zeros_like(vec)
    for l in range(1, label_max + 1):
        mask = l == label
        start = mask.argmax()
        for index in range(start, min(start + width, vec.size)):
            if mask[index]:
                ret[index] = True
    return ret

def _sharpen_array(arr, width):
    if arr.ndim == 1:
        return _sharpen_kernel(arr, width)
    else:
        buf = np.empty(arr.shape, bool)
        for num, elem in enumerate(arr):
            buf[num] = _sharpen_array(elem, width)
        return buf

def _sharpen_packed(args):
    return _sharpen_array(*args)

def _sharpen_parallel(arr, width, n_proc, verbose):
    with mp.Pool(processes=n_proc) as pool:
        args = ((elem, width) for elem in arr)
        buf = np.empty(arr.shape, bool)
        with tqdm.tqdm(args, total=len(arr), disable=not verbose) as pbar:
            for num, ret in enumerate(pool.imap(_sharpen_packed, pbar)):
                buf[num] = ret
    return buf

def sharpen(arr, width, verbose=False):
    arr = np.asarray(arr)
    if arr.ndim == 1 or len(arr) == 1:
        return _sharpen_array(arr, width)
    else:
        n_proc = min(mp.cpu_count() - 1, len(arr))
        return _sharpen_parallel(arr, width, n_proc, verbose)

# binarize -> _binarize_array -> _binarize_kernel
def _binarize_kernel(vec, dev):
    # gaussian approximation of background
    center, std = canal.image.esti.statistic_background(vec, 512)
    centered = vec - center
    return centered > dev * std

def _binarize_array(arr, dev):
    if arr.ndim == 1:
        return _binarize_kernel(arr, dev)
    else:
        buf = np.empty(arr.shape, bool)
        for num, elem in enumerate(arr):
            buf[num] = _binarize_array(elem, dev)
        return buf

def _binarize_packed(args):
    return _binarize_array(*args)

def _binarize_parallel(arr, dev, n_proc, verbose):
    with mp.Pool(processes=n_proc) as pool:
        args = ((elem, dev) for elem in arr)
        buf = np.empty(arr.shape, bool)
        with tqdm.tqdm(args, total=len(arr), disable=not verbose) as pbar:
            for num, ret in enumerate(pool.imap(_binarize_packed, pbar)):
                buf[num] = ret
    return buf

def binarize(arr, dev, verbose=False):
    """
    Creates binarized signals of the input. The threshold would be dev *
    the standard deviation of each signal.

    Parameters
    ----------
    arr: ndarray
    dev: float
    verbose: bool, optional

    Returns
    -------
    binarized: ndarray
    """
    arr = np.asarray(arr)
    if arr.ndim == 1 or len(arr) == 1:
        return _binarize_array(arr, dev)
    else:
        n_proc = min(mp.cpu_count() - 1, len(arr))
        return _binarize_parallel(arr, dev, n_proc, verbose)

def _normalize_kernel(vec, width):
    # baseline estimation
    base_esti = canal.local.minimum(vec, width)
    base_blur = scipy.ndimage.filters.gaussian_filter1d(base_esti, width)
    flat = vec / base_blur - 1

    # gaussian approximation of background noise
    center, std = canal.image.esti.statistic_background(flat, 512)
    return flat - center

def _normalize_array(arr, width):
    if arr.ndim == 1:
        return _normalize_kernel(arr, width)
    else:
        buf = np.empty(arr.shape)
        for num, elem in enumerate(arr):
            buf[num] = _normalize_array(elem, width)
        return buf

def _normalize_packed(args):
    return _normalize_array(*args)

def _normalize_parallel(arr, width, n_proc, verbose):
    with mp.Pool(processes=n_proc) as pool:
        args = ((elem, width) for elem in arr)
        buf = np.empty(arr.shape)
        with tqdm.tqdm(args, total=len(arr), disable=not verbose) as pbar:
            for num, ret in enumerate(pool.imap(_normalize_packed, pbar)):
                buf[num] = ret
    return buf

def normalize(arr, width, verbose=False):
    """
    Calculates the base-line of the signals based on the width in which
    the minimum values would be searched for.

    Parameters
    ----------
    arr: ndarray
    width: int
    verbose: bool, optional

    Returns
    -------
    normalized: ndarray
    """
    arr = np.asarray(arr)
    if arr.ndim == 1 or len(arr) == 1:
        return _normalize_array(arr, width)
    else:
        n_proc = min(mp.cpu_count() - 1, len(arr))
        return _normalize_parallel(arr, width, n_proc, verbose)