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

import mpmath as ap
import numpy as np

class mparray(np.ndarray):
    """
    mparray(shape, ctx=ap.mpf, order=None)

    Provides a convenient view on arrays of arbitrary precision 
    floating-point numbers.

    Versus a regular NumPy array of type `object (``ap.mpf``)`, this
    class adds the following functionality:
      1) assigned values automatically type-checked and converted to 
         the array's context where needed, so provides homogeneity
      2) comparison operators are aware of the context's (interval) precision,
         while some funtions have precision parameters
      3) representation of mparray has the same form of numpy's, except it
         hides it's dtype (obviously being `object`) and shows it's context
    mparrays should be created using `numeric.asmparray`, rather than 
    this constructor directly.

    For more information, refer to the `mpmath` module and examine the
    the methods and attributes of an array.

    Parameters
    ----------
    (for the __new__ method; see Notes below)

    shape : tuple of ints
        Shape of created array.
    ctx : context, optional
        Mpmath context object.
    order : {'C', 'F'}, optional
        Row-major (C-style) or column-major (Fortran-style) order.

    Attributes
    ----------
    T : ndarray
        Transpose of the array.
    data : buffer
        The array's elements, in memory.
    ctx : context object
        Describes the context of the elements in the array.
    flags : dict
        Dictionary containing information related to memory use, e.g.,
        'C_CONTIGUOUS', 'OWNDATA', 'WRITEABLE', etc.
    flat : numpy.flatiter object
        Flattened version of the array as an iterator.  The iterator
        allows assignments, e.g., ``x.flat = 3`` (See `ndarray.flat` for
        assignment examples; TODO).
    imag : ndarray
        Imaginary part of the array.
    real : ndarray
        Real part of the array.
    size : int
        Number of elements in the array.
    itemsize : int
        The memory use of each array element in bytes (8 for object).
    nbytes : int
        The total number of bytes required to store the array data,
        i.e., ``itemsize * size``.
    ndim : int
        The array's number of dimensions.
    shape : tuple of ints
        Shape of the array.
    strides : tuple of ints
        The step-size required to move from one element to the next in
        memory. For example, a contiguous ``(3, 4)`` array of type
        ``int16`` in C-order has strides ``(8, 2)``.  This implies that
        to move from element to element in memory requires jumps of 2 bytes.
        To move from row-to-row, one needs to jump 8 bytes at a time
        (``2 * 4``).
    ctypes : ctypes object
        Class containing properties of the array needed for interaction
        with ctypes.
    base : ndarray
        If the array is a view into another array, that array is its `base`
        (unless that array is also a view).  The `base` array is where the
        array data is actually stored.

    See Also
    --------
    zeros : Create an array, each element of which is zero.
    empty : Create an array, but fills it with None (i.e., it contains
        "garbage"). Homogeneity would NOT be guaranteed in subsequent
        operations. It's the user's responsibility to fill the every
        elements with valid values (arbitrary precision numbers).

    Examples
    --------
    These examples illustrate the low-level `mparray` constructor.  Refer
    to the `See Also` section above for easier ways of constructing an
    mparray.

    >>> mparray(shape=(2,2))
    array([[ nan,  nan],
           [ nan,  nan]])   # do not use it

    """
    def __new__(subtype, shape, ctx=ap.mpf, order=None):
        obj = np.ndarray.__new__(subtype, shape, 'O', order = order)
        obj._ctx = ctx
        return obj

    @property
    def ctx(self):
        return self._ctx

    def __array_finalize__(self, obj):
        if obj is not None:
            self._ctx = getattr(obj, 'ctx', ap.mpf)

    #def __array_wrap__(self, obj, context=None):
    #    return np.ndarray.__array_wrap__(self, obj, context)

    #@property
    #def __array_priority__(self):
    #    return 1.0

    def __setitem__(self, index, obj):
        subarray = np.ndarray.__getitem__(self, index)
        objarray = wrappers.mpmath.numeric.asmparray(obj, self.ctx)
        if not isinstance(subarray, np.ndarray):
            if objarray.size > 1:
                raise ValueError(('could not broadcast input array from shape '
                                  '{} into shape (,)').format(objarray.shape))
            elif objarray.size == 1:
                c = objarray.flat[0]
        else:
            c = objarray
        return np.ndarray.__setitem__(self, index, c)

    def __repr__(self):
        return wrappers.mpmath.numeric.array_repr(self)

    def __eq__(self, obj):
        a1 = wrappers.mpmath.numeric.asmparray(self, ap.iv.mpf)
        a2 = wrappers.mpmath.numeric.asmparray(obj, ap.iv.mpf)
        d = a2 - a1
        return np.array([0 in e for e in d.flat]).reshape(d.shape)

import wrappers.mpmath.numeric