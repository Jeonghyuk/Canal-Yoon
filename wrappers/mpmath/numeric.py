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

__all__ = ['empty', 'empty_like', 'zeros', 'zeros_like', 'ones', 'ones_like',
           'full', 'full_like', 'asmparray', 'array_repr', 'identity',
           'array_equal', 'array_equiv', 'arange']

import mpmath as ap
import numpy as np
import itertools

def empty(shape, ctx=None, order='C'):
    """
    empty(shape, ctx=None, order='C')

    Return a new array of given shape and type, without initializing entries.

    Parameters
    ----------
    shape : int or tuple of int
        Shape of the empty array
    ctx : context, optional
        Desired output context. Default is ``ap.mpf``.
    order : {'C', 'F'}, optional
        Whether to store multi-dimensional data in row-major
        (C-style) or column-major (Fortran-style) order in
        memory.

    Returns
    -------
    out : mparray
        Array of uninitialized (arbitrary) data of the given shape, ctx, and
        order.  Object arrays will be initialized to None.

    See Also
    --------
    empty_like, zeros, ones

    Notes
    -----
    `empty`, unlike `zeros`, does not set the array values to zero,
    and may therefore be marginally faster.  On the other hand, it requires
    the user to manually set all the values in the array, and should be
    used with caution.

    Examples
    --------
    >>> empty([2, 2])
    array([[ nan,  nan],
           [ nan,  nan]])

    >>> empty([2, 2], ctx=ap.iv.mpf)
    array([[None, None],
           [None, None]], ctx=interval)

    """
    if ctx is None:
        ctx = ap.mpf
    return mparray(shape, ctx, order)

def empty_like(a, ctx=None, order='A'):
    """
    empty_like(a, ctx=None, order='A')

    Return a new array with the same shape and type as a given array.

    Parameters
    ----------
    a : array_like
        The shape and data-type of `a` define these same attributes of the
        returned array.
    ctx : context, optional
        Overrides the context of the result.
    order : {'C', 'F', or 'A'}, optional
        Overrides the memory layout of the result. 'C' means C-order,
        'F' means F-order, 'A' means 'F' if ``a`` is Fortran contiguous,
        'C' otherwise. 'K' means match the layout of ``a`` as closely
        as possible.

    Returns
    -------
    out : mparray
        Array of uninitialized (arbitrary) data with the same
        shape and type as `a`.

    See Also
    --------
    ones_like : Return an array of ones with shape and type of input.
    zeros_like : Return an array of zeros with shape and type of input.
    empty : Return a new uninitialized array.
    ones : Return a new array setting values to one.
    zeros : Return a new array setting values to zero.

    Notes
    -----
    This function does *not* initialize the returned array; to do that use
    `zeros_like` or `ones_like` instead.  It may be marginally faster than
    the functions that do set the array values.

    Examples
    --------
    >>> a = ([1,2,3], [4,5,6])                         # a is array-like
    >>> empty_like(a)
    array([[ nan,  nan,  nan],
           [ nan,  nan,  nan]])
    >>> a = np.array([[1., 2., 3.],[4.,5.,6.]])
    >>> empty_like(a)
    array([[ nan,  nan,  nan],
           [ nan,  nan,  nan]])
    """
    a = asmparray(a, ctx)
    if order == 'A':
        if a.flags.f_contiguous:
            order = 'F'
        else:
            order = 'C'
    return mparray(a.shape, a.ctx, order)

def zeros(shape, ctx=None, order='C'):
    """
    Return a new array of given shape and type, filled with zeros.

    Parameters
    ----------
    shape : int or sequence of ints
        Shape of the new array, e.g., ``(2, 3)`` or ``2``.
    ctx : context, optional
        The desired context for the array, e.g., ``mpmath.iv.mpf``.
        Default is ``mpmath.mpf``.
    order : {'C', 'F'}, optional
        Whether to store multidimensional data in C- or Fortran-contiguous
        (row- or column-wise) order in memory.

    Returns
    -------
    out : mparray
        Array of zeros with the given shape, ctx, and order.

    See Also
    --------
    zeros_like : Return an array of zeros with shape and type of input.
    ones_like : Return an array of ones with shape and type of input.
    empty_like : Return an empty array with shape and type of input.
    ones : Return a new array setting values to one.
    empty : Return a new uninitialized array.

    Examples
    --------
    >>> zeros(5)
    array([ 0.,  0.,  0.,  0.,  0.])

    >>> zeros((2, 1))
    array([[ 0.],
           [ 0.]])

    >>> s = (2,2)
    >>> zeros(s)
    array([[ 0.,  0.],
           [ 0.,  0.]])

    """
    a = empty(shape, ctx, order)
    np.copyto(a, a.ctx(0), casting='unsafe')
    return a

def zeros_like(a, ctx=None, order='A'):
    """
    Return a new array of given shape and type, filled with zeros.

    Parameters
    ----------
    a : array_like
        The shape and data-type of `a` define these same attributes of
        the returned array.
    ctx : context, optional
        Overrides the context of the result.
    order : {'C', 'F', 'A', or 'K'}, optional
        Overrides the memory layout of the result. 'C' means C-order,
        'F' means F-order, 'A' means 'F' if `a` is Fortran contiguous,
        'C' otherwise. 'K' means match the layout of `a` as closely
        as possible.
    subok : bool, optional.
        If True, then the newly created array will use the sub-class
        type of 'a', otherwise it will be a base-class array. Defaults
        to True.

    Returns
    -------
    out : mparray
        Array of zeros with the same shape and type as `a`.

    See Also
    --------
    ones_like : Return an array of ones with shape and type of input.
    empty_like : Return an empty array with shape and type of input.
    zeros : Return a new array setting values to zero.
    ones : Return a new array setting values to one.
    empty : Return a new uninitialized array.

    Examples
    --------
    >>> x = arange(6)
    >>> x = x.reshape((2, 3))
    >>> x
    array([[ 0.,  1.,  2.],
           [ 3.,  4.,  5.]])
    >>> zeros_like(x)
    array([[ 0.,  0.,  0.],
           [ 0.,  0.,  0.]])
    >>> y = arange(3, ctx=ap.iv.mpf)
    >>> y
    array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]], ctx=interval)
    >>> zeros_like(y)
    array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], ctx=interval)

    """
    res = empty_like(a, ctx=ctx, order=order)
    np.copyto(res, res.ctx(0), casting='unsafe')
    return res

def ones(shape, ctx=None, order='C'):
    """
    Return a new array of given shape and type, filled with ones.

    Parameters
    ----------
    shape : int or sequence of ints
        Shape of the new array, e.g., ``(2, 3)`` or ``2``.
    ctx : context, optional
        The desired context for the array, e.g., `numpy.int8`.  Default is
        `numpy.float64`.
    order : {'C', 'F'}, optional
        Whether to store multidimensional data in C- or Fortran-contiguous
        (row- or column-wise) order in memory.

    Returns
    -------
    out : mparray
        Array of ones with the given shape, ctx, and order.

    See Also
    --------
    zeros, ones_like

    Examples
    --------
    >>> ones(5)
    array([ 1.,  1.,  1.,  1.,  1.])

    >>> ones((2, 1))
    array([[ 1.],
           [ 1.]])

    >>> s = (2,2)
    >>> ones(s)
    array([[ 1.,  1.],
           [ 1.,  1.]])

    """
    a = empty(shape, ctx, order)
    np.copyto(a, a.ctx(1), casting='unsafe')
    return a

def ones_like(a, ctx=None, order='K', subok=True):
    """
    Return an array of ones with the same shape and type as a given array.

    Parameters
    ----------
    a : array_like
        The shape and data-type of `a` define these same attributes of
        the returned array.
    ctx : context, optional
        Overrides the context of the result.
    order : {'C', 'F', 'A', or 'K'}, optional
        Overrides the memory layout of the result. 'C' means C-order,
        'F' means F-order, 'A' means 'F' if `a` is Fortran contiguous,
        'C' otherwise. 'K' means match the layout of `a` as closely
        as possible.
    subok : bool, optional.
        If True, then the newly created array will use the sub-class
        type of 'a', otherwise it will be a base-class array. Defaults
        to True.

    Returns
    -------
    out : mparray
        Array of ones with the same shape and type as `a`.

    See Also
    --------
    zeros_like : Return an array of zeros with shape and type of input.
    empty_like : Return an empty array with shape and type of input.
    zeros : Return a new array setting values to zero.
    ones : Return a new array setting values to one.
    empty : Return a new uninitialized array.

    Examples
    --------
    >>> x = arange(6)
    >>> x = x.reshape((2, 3))
    >>> x
    array([[ 0.,  1.,  2.],
           [ 3.,  4.,  5.]])
    >>> ones_like(x)
    array([[ 1.,  1.,  1.],
           [ 1.,  1.,  1.]])
    >>> y = arange(3, ctx=ap.iv.mpf)
    >>> y
    array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]], ctx=interval)
    >>> ones_like(y)
    array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]], ctx=interval)

    """
    res = empty_like(a, ctx=ctx, order=order)
    np.copyto(res, res.ctx(1), casting='unsafe')
    return res

def full(shape, fill_value, ctx=None, order='C'):
    """
    Return a new array of given shape and type, filled with `fill_value`.

    Parameters
    ----------
    shape : int or sequence of ints
        Shape of the new array, e.g., ``(2, 3)`` or ``2``.
    fill_value : scalar
        Fill value.
    ctx : context, optional
        The desired context for the array  The default, `None`, means
        ``ap.mpf`` without provided fill_value of ``ap.iv.mpf``.
    order : {'C', 'F'}, optional
        Whether to store multidimensional data in C- or Fortran-contiguous
        (row- or column-wise) order in memory.

    Returns
    -------
    out : mparray
        Array of `fill_value` with the given shape, ctx, and order.

    See Also
    --------
    zeros_like : Return an array of zeros with shape and type of input.
    ones_like : Return an array of ones with shape and type of input.
    empty_like : Return an empty array with shape and type of input.
    full_like : Fill an array with shape and type of input.
    zeros : Return a new array setting values to zero.
    ones : Return a new array setting values to one.
    empty : Return a new uninitialized array.

    Examples
    --------
    >>> full((2, 2), ap.inf)
    array([[ inf,  inf],
           [ inf,  inf]])
    >>> full((2, 2), 10)
    array([[ 10.,  10.],
           [ 10.,  10.]])

    """
    if ctx is None:
        ctx = ap.iv.mpf if isinstance(fill_value, ap.iv.mpf) else ap.mpf
    a = empty(shape, ctx, order)
    np.copyto(a, ctx(fill_value), casting='unsafe')
    return a

def full_like(a, fill_value, ctx=None, order='A'):
    """
    Return a full array with the same shape and type as a given array.

    Parameters
    ----------
    a : array_like
        The shape and data-type of `a` define these same attributes of
        the returned array.
    fill_value : scalar
        Fill value.
    ctx : context, optional
        Overrides the context of the result.
    order : {'C', 'F', or 'A'}, optional
        Overrides the memory layout of the result. 'C' means C-order,
        'F' means F-order, 'A' means 'F' if `a` is Fortran contiguous,
        'C' otherwise. 'K' means match the layout of `a` as closely
        as possible.

    Returns
    -------
    out : mparray
        Array of `fill_value` with the same shape and context as `a`.

    See Also
    --------
    zeros_like : Return an array of zeros with shape and type of input.
    ones_like : Return an array of ones with shape and type of input.
    empty_like : Return an empty array with shape and type of input.
    zeros : Return a new array setting values to zero.
    ones : Return a new array setting values to one.
    empty : Return a new uninitialized array.
    full : Fill a new array.

    Examples
    --------
    >>> x = np.arange(6, dtype=np.int)
    >>> full_like(x, 1)
    array([ 1.,  1.,  1.,  1.,  1.,  1.])
    >>> y = arange(6, ctx=ap.iv.mpf).reshape((2, 3))
    >>> full_like(y, 1))
    array([[[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],
           [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]], ctx=interval)

    """
    res = empty_like(a, ctx=ctx, order=order)
    np.copyto(res, res.ctx(fill_value), casting='unsafe')
    return res

def asmparray(a, ctx=None, precision=None, order=None):
    """
    Convert the input to an mparray.

    Parameters
    ----------
    a : array_like
        Input data, in any form that can be converted to an array.  This
        includes scalars, lists, lists of tuples, tuples, tuples of tuples,
        tuples of lists, and ndarrays.
    ctx : context, optional
        By default, the context object is inferred from the input data.
    precision : int, optional
        By default, the precision is inferred from the input data.
    order : {'C', 'F'}, optional
        Whether to use row-major (C-style) or column-major
        (Fortran-style) memory representation.  Defaults to 'C'.

    Returns
    -------
    out : mparray
        Array interpretation of `a`.  If `a` is an mparray while ctx and
        precision are not set, it is returned as-is and no copy is performed.

    See Also
    --------
    asarray : Similar function which always returns ndarrays.
    ascontiguousarray : Convert input to a contiguous array.
    asfarray : Convert input to a floating point ndarray.
    asfortranarray : Convert input to an ndarray with column-major
                     memory order.
    asarray_chkfinite : Similar function which checks input for NaNs and
                        Infs.
    fromiter : Create an array from an iterator.
    fromfunction : Construct an array by executing a function on grid
                   positions.

    Examples
    --------
    Convert a list into an array:
    >>> a = [1, 2]
    >>> asmparray(a)
    array([ 1.,  2.])

    """
    if (isinstance(a, mparray) and (ctx is None or ctx == a.ctx) and
        precision is None):
        return a
    s = np.array(a, copy=False, order=order, subok=True)
    if ctx is None:
        if s.dtype == 'O':
            check = [isinstance(elem, ap.iv.mpf) for elem in s.flat]
            ctx = ap.iv.mpf if np.asarray(check).any() else ap.mpf
        else:
            ctx = ap.mpf
    try:
        if precision is not None:
            prev_prec = (ap.mp.prec, ap.iv.prec)
            ap.mp.prec, ap.iv.prec = [precision] * 2
        if np.issubdtype(s.dtype, np.integer):
            p = np.asarray([ctx(int(elem)) for elem in s.flat])
        elif np.issubdtype(s.dtype, np.inexact):
            p = np.asarray([ctx(elem) for elem in s.flat])
        else:
            p = np.asarray([ctx(str(elem)) for elem in s.flat])
    finally:
        if 'prev_prec' in locals():
            ap.mp.prec, ap.iv.prec = prev_prec
    v = p.reshape(s.shape).view(mparray)
    v._ctx = ctx
    return v

_ctxless = [ap.mpf]
_ctxstring = {ap.mpf: 'number', ap.iv.mpf: 'interval'}
ap.mp.pretty = True
ap.iv.pretty = True

def array_repr(arr, max_line_width=None, precision=None, suppress_small=None):
    """
    Return the string representation of an array.

    Parameters
    ----------
    arr : mparray
        Input array.
    max_line_width : int, optional
        The maximum number of columns the string should span. Newline
        characters split the string appropriately after array elements.
    precision : int, optional
        Floating point precision. Default is the current printing precision
        (usually 8), which can be altered using `set_printoptions`.
    suppress_small : bool, optional
        Represent very small numbers as zero, default is False. Very small
        is defined by `precision`, if the precision is 8 then
        numbers smaller than 5e-9 are represented as zero.

    Returns
    -------
    string : str
      The string representation of an array.

    See Also
    --------
    array_str, array2string, set_printoptions

    Examples
    --------
    >>> array_repr(asmparray([1,2]))
    'array([ 1.,  2.])'
    >>> array_repr(asmparray([], ap.iv.mpf))
    'array([], ctx=interval)'

    """
    if arr.__class__ is not mparray:
        raise ValueError('input should be mparray class')
    cName = "array"

    if arr.size > 0 or arr.shape == (0,):
        reparr = np.asfarray(arr) if arr.ctx == ap.mpf else arr
        lst = np.array2string(reparr, max_line_width, precision, 
                              suppress_small, ', ', "array(")
    else:  # show zero-length shape unless it is (0,)
        lst = "[], shape=%s" % (repr(arr.shape),)

    skipctx = (arr.ctx in _ctxless) and arr.size > 0

    if skipctx:
        return "%s(%s)" % (cName, lst)
    else:
        return cName + "(%s, ctx=%s)" % (lst, _ctxstring[arr.ctx])

def identity(n, ctx=None):
    """
    Return the identity array.

    The identity array is a square array with ones on
    the main diagonal.

    Parameters
    ----------
    n : int
        Number of rows (and columns) in `n` x `n` output.
    ctx : context, optional
        Context of the output.  Defaults to ``float``.

    Returns
    -------
    out : mparray
        `n` x `n` array with its main diagonal set to one,
        and all other elements 0.

    Examples
    --------
    >>> identity(3)
    array([[ 1.,  0.,  0.],
           [ 0.,  1.,  0.],
           [ 0.,  0.,  1.]])

    """
    from .twodim_base import eye
    return eye(n, ctx=ctx)

def array_equal(a1, a2, precision=None):
    """
    True if two arrays have the same shape and elements, False otherwise.

    Parameters
    ----------
    a1, a2 : array_like
        Input arrays.
    precision: integer
        precision used on comparing.

    Returns
    -------
    b : bool
        Returns True if the arrays are equal.

    See Also
    --------
    allclose: Returns True if two arrays are element-wise equal within a
              tolerance.
    array_equiv: Returns True if input arrays are shape consistent and all
                 elements equal.

    Examples
    --------
    >>> array_equal([1, 2], [1, 2])
    True
    >>> array_equal([1, 2], [1, 2, 3])
    False
    >>> array_equal([1, 2], [1, 4])
    False
    >>> array_equal(0.1 * 3, 0.3)
    False
    >>> array_equal(0.1 * 3, 0.3, precision=50)
    True
    
    """
    try:
        a1 = asmparray(a1, ap.iv.mpf, precision=precision)
        a2 = asmparray(a2, ap.iv.mpf, precision=precision)
    except:
        return False
    if a1.shape != a2.shape:
        return False
    return bool(np.asarray(a1 == a2).all())

def array_equiv(a1, a2, precision=None):
    """
    Returns True if input arrays are shape consistent and all elements equal.

    Shape consistent means they are either the same shape, or one input array
    can be broadcasted to create the same shape as the other one.

    Parameters
    ----------
    a1, a2 : array_like
        Input arrays.

    Returns
    -------
    out : bool
        True if equivalent, False otherwise.

    Examples
    --------
    >>> array_equiv([1, 2], [1, 2])
    True
    >>> array_equiv([1, 2], [1, 3])
    False

    Showing the shape equivalence:
    >>> array_equiv([1, 2], [[1, 2], [1, 2]])
    True
    >>> array_equiv([1, 2], [[1, 2, 1, 2], [1, 2, 1, 2]])
    False
    >>> array_equiv([1, 2], [[1, 2], [1, 3]])
    False
    
    Showing the precision dependency:
    >>> array_equiv([0.1 * 3] * 2, 0.3)
    False
    >>> array_equiv([0.1 * 3] * 2, 0.3, precision=50)
    True

    """
    try:
        a1 = asmparray(a1, ap.iv.mpf, precision=precision)
        a2 = asmparray(a2, ap.iv.mpf, precision=precision)
    except:
        return False
    try:
        np.broadcast(a1, a2)
    except:
        return False
    return bool(np.asarray(a1 == a2).all())

def arange(start=None, stop=None, step=None, ctx=None):
    """
    Return evenly spaced values within a given interval.

    Values are generated within the half-open interval ``[start, stop)``
    (in other words, the interval including `start` but excluding `stop`).
    For integer arguments the function is equivalent to the Python built-in
    `range <http://docs.python.org/lib/built-in-funcs.html>`_ function,
    but returns an mparray rather than a list.

    When using a non-integer step, such as 0.1, the results will often not
    be consistent.  It is better to use high precision for these cases.

    Parameters
    ----------
    start : number, optional
        Start of interval.  The interval includes this value.  The default
        start value is 0.
    stop : number
        End of interval.  The interval does not include this value, except
        in some cases where `step` is not an integer and floating point
        round-off affects the length of `out`.
    step : number, optional
        Spacing between values.  For any output `out`, this is the distance
        between two adjacent values, ``out[i+1] - out[i]``.  The default
        step size is 1.  If `step` is specified, `start` must also be given.
    ctx : context, optional
        The context of the output array.  If `ctx` is not given, infer
        the data context from the other input arguments.

    Returns
    -------
    arange : mparray
        Array of evenly spaced values.

        For floating point arguments, the length of the result is
        ``ceil((stop - start)/step)``.  Because of floating point overflow,
        this rule may result in the last element of `out` being greater
        than `stop`.

    See Also
    --------
    linspace : Evenly spaced numbers with careful handling of endpoints.
    ogrid: Arrays of evenly spaced numbers in N-dimensions.
    mgrid: Grid-shaped arrays of evenly spaced numbers in N-dimensions.

    Examples
    --------
    >>> arange(3)
    array([ 0.,  1.,  2.])
    >>> arange(3,7)
    array([ 3.,  4.,  5.,  6.])
    >>> arange(3,7,2)
    array([ 3.,  5.])
    >>> np.arange(0, 2.1, 0.3)
    array([ 0. ,  0.3,  0.6,  0.9,  1.2,  1.5,  1.8,  2.1]) # not consistent
    >>> arange(0, 2.1, 0.3)
    array([ 0. ,  0.3,  0.6,  0.9,  1.2,  1.5,  1.8])       # as expected

    """
    if start is None:
        raise TypeError("Required argument 'start' (pos 1) not found")
    builtin = ((start is None or isinstance(start, int)) and
               (stop is None or isinstance(stop, int)) and
               (step is None or isinstance(step, int)))
    if builtin:
        if step is None:
            v = range(start) if stop is None else range(start, stop)
        else:
            v = range(0, start, step) if stop is None else range(start, stop, step)
    else:
        if stop is None:
            begin, end = ap.mpf(0), ap.mpf(str(start))
        else:
            begin, end = ap.mpf(str(start)), ap.mpf(str(stop))
        inc = ap.mpf(1) if step is None else ap.mpf(str(step))
        v = []
        biv, siv = ap.iv.mpf(begin), ap.iv.mpf(inc)
        for num in itertools.count():
            ni = biv + siv * num
            if ni < end:
                ne = begin + inc * num
                v.append(ne)
            else:
                break
    if ctx is None:
        ctx = ap.mpf
    return asmparray([ctx(elem) for elem in v])

from wrappers.mpmath.defmparray import mparray