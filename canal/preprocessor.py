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
import multiprocessing as mp

import nibabel as nib
import skimage.feature
import json
import os.path

class Splicer:
    def splice(loaders, writer, filename, buffername=None):
        shapes = [elem.shape for elem in loaders]
        positions = (0,) + tuple(np.cumsum([len(elem) for elem in loaders]))
        new_shape = (positions[-1],) + shapes[0][1:]
        dtypes = [elem.dtype for elem in loaders]
        new_dtype = dtypes[0]

        if buffername is None:  # use memory
            buf = np.empty(new_shape, new_dtype)
        else:   # use memmap whose filename is buffername
            if os.path.exists(buffername):
                raise ValueError('Buffer file already in use')
            buf = np.memmap(buffername, dtype=new_dtype,
                            mode='w+', shape=new_shape)

        try:    # catch buffer
            n_work = len(loaders)
            fill_len = len(str(n_work))
            progress = 'Loading: {{:0{}}}/{}'.format(fill_len, n_work)
            for num, (begin, end) in enumerate(zip(positions[:-1],
                                                   positions[1:])):
                print(progress.format(num))
                buf[begin:end] = loaders[num].load()
            writer.dump(buf, filename)
        finally:
            del buf
            if buffername is not None:
                os.remove(buffername)

def _permutation(axes_to, axes_from):
    if set(axes_to) != set(axes_from):
        raise Exception('Different axes')

    ret = (axes_from.find(axis) for axis in axes_to)
    return tuple(ret)
    

def _print_progress(string, show):
    _erase_line = '\x1b[2K'
    if show:
        print(_erase_line + string, end = '\r')

def _ras_to_affine(r=None, a=None, s=None):
    def get_unit(axis):
        direction = np.array([elem in axis for elem in 'xyz'], int)
        signed = -direction if '-' in axis else direction
        return signed
        
    axis_strings = (r, a, s)
    axis_vectors = [get_unit(elem) if elem != None else None
                    for elem in axis_strings]
    
    # validate input
    is_none = [elem is None for elem in axis_vectors]
    num_none = sum(is_none)
    if num_none > 1:    # guess failed
        raise Exception('Unable to guess: need more information.')
    elif num_none == 1: # guess None element
        guess_axis = sum(is_none * np.arange(3))
        axis_vectors[guess_axis] = np.cross(axis_vectors[guess_axis - 2],
                                            axis_vectors[guess_axis - 1])
    else:   # check whether the inputs don't conflict with each other
        if not np.all(axis_vectors[2] == np.cross(axis_vectors[0],
                                                  axis_vectors[1])):
            raise Exception('Invalid input: condition np.cross(r, a) == s not met.') 
    
    affine = np.eye(4)
    affine[:3, :3] = np.array(axis_vectors)
    return affine
    
def _offset_to_affine(xyz_offset):
    affine = np.eye(4)
    affine[:3, 3] += xyz_offset
    return affine
    
def _translations_by_fft(series, tpl_image, tpl_origin, stack_axis,
                         processes):
    if stack_axis < 0:  # compatible with python protocol
        stack_axis = series.ndim - stack_axis
    
    # get window to make data which you compare with
    def get_window(begin, shape, stack_axis):
        end = np.array(begin) + shape
        window = [(b, e) for b, e in zip(begin, end)]
        window[stack_axis] = (0, -1)
        return window
    
    # prepare stack to compare
    z_window, y_window, x_window = get_window(tpl_origin, tpl_image.shape,
                                              stack_axis - 1)
    stack = series[:, z_window[0]:z_window[1],
                   y_window[0]:y_window[1],
                   x_window[0]:x_window[1]].max(axis = stack_axis)
    tpl_slice = tpl_image.max(axis = stack_axis - 1)
                   
    # compare images and template               
    time_size = len(series)
    with mp.Pool(processes = processes) as pool:
        args = ((stack[time], tpl_slice) for time in range(time_size))
        plane_offsets = np.array(pool.map(_translation_by_fft_margs, args))
    
    # convert offsets of planes to those in volume (2d -> 3d)
    def get_volume_offsets(plane_offsets, stack_axis):    
        offsets = list(zip(*plane_offsets))
        offsets.insert(stack_axis, (0,) * len(plane_offsets))
        return np.array(offsets).T
    
    # map offsets to parent space
    local_offsets = get_volume_offsets(plane_offsets, stack_axis - 1)
    offsets = local_offsets + np.array(tpl_origin)[np.newaxis, :]
    return offsets
    
def _translation_by_landmarks_margs(args):
    return translation_by_landmarks(*args)

def registration(dst, src):
	"""
	Registration under rotation and translation based on fft.

	References
	----------
	[1] Reddy, B. Srinivasa, and Biswanath N. Chatterji.
	    "An FFT-based technique for translation, rotation, and scale-invariant
	    image registration."
	    IEEE transactions on image processing 5.8 (1996): 1266-1271.
	"""
	return translation(dst, src)

def _translation_by_fft_margs(args):
    return translation(*args)

def translation_by_fft(dst, src, crit = 0.02, distribution = False):
	"""
	Returns displacement of translation.

	Parameters
	----------
	dst: array of shape (R, C)
	src: array of shape (R, C)
	crit: float
	crit must be in a range [0, 1].

	Returns
	-------
	row: integer
	col: integer
	"""

	row_size, col_size = src.shape
	cross = np.fft.fft2(dst) * np.fft.fft2(src).conj()
	delta = np.fft.ifft2(cross / np.abs(cross)).real

	if distribution == True:
		return delta
	
	row, col = np.unravel_index(delta.argmax(), delta.shape)
	max_val = delta[row, col]
	if max_val < crit:
		raise Exception('Criterion not met: {} (< {})'.format(max_val, crit))

	row = row if row < row_size // 2 else row - row_size
	col = col if col < col_size // 2 else col - col_size
	return (row, col)
    
def rotation(dst, src, crit = 0.03):
	row_size, col_size = src.shape
	min_size = min(row_size, col_size)

	row_min, col_min = (row_size - min_size) // 2, (col_size - min_size) // 2
	dst_sub = dst[row_min:row_min + min_size, col_min:col_min + min_size]
	src_sub = src[row_min:row_min + min_size, col_min:col_min + min_size]

	cross = np.fft.fft2(np.abs(pfft(dst_sub))) * np.fft.fft2(np.abs(pfft(src_sub))).conj()
	delta = np.abs(np.fft.ifft2(cross / np.abs(cross)))	

	radius, angle = np.unravel_index(delta.argmax(), delta.shape)
	max_val = delta[radius, angle]
	if max_val < crit:
		raise Exception('Criterion not met: {} (< {})'.format(max_val, crit))

	angle /= np.pi / 2 / min_size
	return (radius, angle)

def frft(vec, alpha):
	"""
	FRFT: FRactional Fourier Transform.
	Optimization of the following code.

	acc = np.zeros(vec.size, dtype = complex)	
	for k in range(vec.size):
		for n in range(vec.size):
			acc[k] += vec[n] * np.exp(-2j * np.pi * alpha * n * k / vec.size)
	return acc
	"""
	domain = np.arange(-vec.size + 1, vec.size)
	sq = domain * domain
	conv = np.exp(1j * np.pi * alpha / vec.size * sq)
	mul = np.exp(-1j * np.pi * alpha / vec.size * sq[-vec.size:])
	return np.convolve(vec * mul, conv, mode = 'valid') * mul

def pfft(src):
	"""
	Polar Fast Fourier Transform.

	References
	----------
	[1] Averbuch, Amir, et al. 
	    "Fast and accurate polar Fourier transform."
	    Applied and computational harmonic analysis 21.2 (2006): 145-167.
	"""
	row_size, col_size = src.shape
	if row_size != col_size:
		raise Exception('Condition src.shape[0] == src.shape[1] not met')
	sampling = row_size	# (= col_size)

	# Pseudo polar fast fourier transform
	sign = np.array([1 if num % 2 == 0 else -1 for num in range(sampling)])
	## row-wise
	row_ffted = np.empty((sampling * 2, sampling), complex)
	for col in range(sampling):
		row_ffted[:, col] = np.fft.fft(src[:, col] * sign[col], sampling * 2)
	col_range = ((-sampling + 1) // 2, (sampling + 1) // 2)

	row_wise = np.empty(row_ffted.shape, complex)
	for row in range(sampling * 2):
		premul = np.exp(-2j * np.pi * col_range[0] * (row - sampling) / sampling / sampling * np.arange(sampling))
		row_wise[row] = FRFT(row_ffted[row] * premul, (row - sampling) / sampling)

	## col-wise
	col_ffted = np.empty((sampling, sampling * 2), complex)
	for row in range(sampling):
		col_ffted[row] = np.fft.fft(src[row] * sign[row], sampling * 2)
	row_range = ((-sampling + 2) // 2, (sampling + 2) // 2)

	col_wise = np.empty(col_ffted.shape, complex)
	for col in range(sampling * 2):
		premul = np.exp(-2j * np.pi * row_range[0] * (col - sampling) / sampling / sampling * np.arange(sampling))
		col_wise[:, col] = FRFT(col_ffted[:, col] * premul, (col - sampling) / sampling)

	def interp_complex(x, xp, fp, left = None, right = None):
		cplex = np.empty(x.size, complex)
		cplex.real = np.interp(x, xp, fp.real, left, right)
		cplex.imag = np.interp(x, xp, fp.imag, left, right)
		return cplex
		
	# rotate the rays
	## row-wise
	col_domain = 2 / sampling * np.arange(*col_range)
	col_image = np.tan(np.pi / 2 / sampling * np.arange(*col_range))
	row_rotated = np.empty(row_wise.shape, complex)
	for row in range(sampling * 2):
		row_rotated[row] = interp_complex(col_image, col_domain, row_wise[row])

	## col-wise
	row_domain = 2 / sampling * np.arange(*row_range)
	row_image = np.tan(np.pi / 2 / sampling * np.arange(*row_range))
	col_rotated = np.empty(col_wise.shape, complex)
	for col in range(sampling * 2):
		col_rotated[:, col] = interp_complex(row_image, row_domain, col_wise[:, col])

	# circle the squares
	rad_domain = np.arange(-sampling, sampling)
	## row-wise
	col_denoms = np.abs(np.cos(np.pi / 2 / sampling * np.arange(*col_range)))
	row_circled = np.empty(row_rotated.shape, complex)
	for col in range(sampling):
		row_circled[:, col] = interp_complex(col_denoms[col] * rad_domain, rad_domain, row_rotated[:, col])

	## col-wise
	row_denoms = np.abs(np.cos(np.pi / 2 / sampling * np.arange(*row_range)))
	col_circled = np.empty(col_rotated.shape, complex)
	for row in range(sampling):
		col_circled[row] = interp_complex(row_denoms[row] * rad_domain, rad_domain, col_rotated[row])

	# splice the components
	spliced = np.empty((sampling, 4 * sampling), complex)
	spliced[:, :sampling] = row_circled[sampling:, ::-1]
	spliced[:, sampling:sampling * 2] = col_circled[:, sampling:0:-1].T
	spliced[:, sampling * 2:sampling * 3] = row_circled[sampling:0:-1, ::-1]
	spliced[:, sampling * 3:] = col_circled[:, sampling:].T

	return spliced
