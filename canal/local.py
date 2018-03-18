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

def minimum(vec, width):
    size = vec.size
    ret = np.empty_like(vec)
    for time in range(size):
        begin = max(0, time - width)
        end = min(size, time + width + 1)
        ret[time] = np.min(vec[begin:end])
    return ret

def maximum(vec, width):
    size = vec.size
    ret = np.empty_like(vec)
    for time in range(size):
        begin = max(0, time - width)
        end = min(size, time + width + 1)
        ret[time] = np.max(vec[begin:end])
    return ret

def percentile(vec, width, p):
    size = vec.size
    ret = np.empty_like(vec)
    for time in range(size):
        begin = max(0, time - width)
        end = min(size, time + width + 1)
        ret[time] = np.percentile(vec[begin:end], p)
    return ret
    
def std(vec, width):
    """
    local std for large data.
    """
    size = vec.size
    ret = np.empty(size)   
    for time in range(size):
        begin = max(0, time - width)
        end = min(size, time + width + 1)
        ret[time] = np.std(vec[begin:end])
    return ret
    
def _local_std_micro(data, halfwidth):
    """
    local std for small data.
    """       
    stack_shape = (halfwidth * 2 + 1, 
                   data.shape[0] + halfwidth * 2) + data.shape[1:]
    stack = np.empty(stack_shape)
    for index in range(len(stack)):
        stack[index, index:index + data.shape[0]] = data

    result = np.empty(data.shape)
    middle = stack[:, halfwidth * 2:-halfwidth * 2]
    result[halfwidth:-halfwidth] = middle.std(axis = 0)
    
    for index in range(halfwidth):
        result[index] = stack[:index + halfwidth + 1, 
		                      index + halfwidth].std(axis = 0)

    for index in reversed(range(-halfwidth, 0)):
        result[index] = stack[index - halfwidth:,
		                      index - halfwidth].std(axis = 0)

    return result

def sum(vec, width):
    kernel = np.ones(width * 2 + 1, vec.dtype)
    return np.convolve(vec, kernel, mode='same')

def mean(vec, width):
    ksize = width * 2 + 1
    kernel = np.ones(ksize) / ksize
    return np.convolve(vec, kernel, mode='same')
