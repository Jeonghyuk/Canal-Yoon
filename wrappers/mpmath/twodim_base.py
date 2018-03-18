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
import mpmath as ap

def eye(N, M=None, k=0, ctx=ap.mpf):
    """
    Return a 2-D array with ones on the diagonal and zeros elsewhere.

    Parameters
    ----------
    N : int
        Number of rows in the output.
    M : int, optional
        Number of columns in the output. If None, defaults to `N`.
    k : int, optional
        Index of the diagonal: 0 (the default) refers to the main diagonal,
        a positive value refers to an upper diagonal, and a negative value
        to a lower diagonal.
    ctx : context, optional
        Context of the returned array.

    Returns
    -------
    I : mparray of shape (N,M)
        An array where all elements are equal to zero, except for the `k`-th
        diagonal, whose values are equal to one.

    See Also
    --------
    identity : (almost) equivalent function
    diag : diagonal 2-D array from a 1-D array specified by the user.

    Examples
    --------
    >>> eye(2)
    array([[1.0, 0.0],
           [0.0, 1.0]])
    >>> eye(3, k=1)
    array([[0.0, 1.0, 0.0],
           [0.0, 0.0, 1.0],
           [0.0, 0.0, 0.0]])

    """
    if M is None:
        M = N
    m = wrappers.mpmath.numeric.zeros((N, M), ctx=ctx)
    if k >= M:
        return m
    if k >= 0:
        i = k
    else:
        i = (-k) * M
    m[:M-k].flat[i::M+1] = m.ctx(1)
    return m

def diag(v, k=0):
    """
    Extract a diagonal or construct a diagonal array.

    See the more detailed documentation for ``numpy.diagonal`` if you use this
    function to extract a diagonal and wish to write to the resulting array;
    whether it returns a copy or a view depends on what version of numpy you
    are using.

    Parameters
    ----------
    v : array_like
        If `v` is a 2-D array, return a copy of its `k`-th diagonal.
        If `v` is a 1-D array, return a 2-D array with `v` on the `k`-th
        diagonal.
    k : int, optional
        Diagonal in question. The default is 0. Use `k>0` for diagonals
        above the main diagonal, and `k<0` for diagonals below the main
        diagonal.

    Returns
    -------
    out : mparray
        The extracted diagonal or constructed diagonal array.

    See Also
    --------
    diagonal : Return specified diagonals.
    diagflat : Create a 2-D array with the flattened input as a diagonal.
    trace : Sum along diagonals.
    triu : Upper triangle of an array.
    tril : Lower triangle of an array.

    Examples
    --------
    >>> x = arange(9).reshape((3,3))
    >>> x
    array([[0, 1, 2],
           [3, 4, 5],
           [6, 7, 8]])

    >>> diag(x)
    array([0.0, 4.0, 8.0])
    >>> diag(x, k=1)
    array([1.0, 5.0])
    >>> diag(x, k=-1)
    array([3.0, 7.0])

    >>> diag(diag(x))
    array([[0.0, 0.0, 0.0],
           [0.0, 4.0, 0.0],
           [0.0, 0.0, 8.0]])

    """
    v = wrappers.mpmath.numeric.asmparray(v)
    s = v.shape
    if len(s) == 1:
        n = s[0]+abs(k)
        res = wrappers.mpmath.numeric.zeros((n, n), v.ctx)
        if k >= 0:
            i = k
        else:
            i = (-k) * n
        res[:n-k].flat[i::n+1] = v
        return res
    elif len(s) == 2:
        return np.diagonal(v, k)
    else:
        raise ValueError("Input must be 1- or 2-d.")

import wrappers.mpmath.numeric