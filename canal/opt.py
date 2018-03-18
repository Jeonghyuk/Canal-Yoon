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
import itertools

def split(shape, partition, margin):
    """
    Parameters
    ----------
    shape: tuple
    partition: tuple
    margin: tuple

    Returns
    -------
    splitters: list
    setters: list
    getters: list
    """
    borders = [np.linspace(0, s, p + 1).astype(int)
               for s, p in zip(shape, partition)]

    begins = [(e[0],) + tuple(e[1:-1] - m) for e, m in zip(borders, margin)]
    ends = [tuple(e[1:-1] + m) + (e[-1],) for e, m in zip(borders, margin)]
    splitters = [tuple(slice(b, e) for b, e in zip(begin, end))
                for begin, end in zip(itertools.product(*begins),
                                      itertools.product(*ends))]

    begins = [tuple(b[:-1]) for b in borders]
    ends = [tuple(b[1:]) for b in borders]
    setters = [tuple(slice(b, e) for b, e in zip(begin, end))
               for begin, end in zip(itertools.product(*begins),
                                    itertools.product(*ends))]
    getters = [tuple(slice(st.start - sp.start, st.stop - sp.start)
                     for st, sp in zip(setter, splitter))
               for setter, splitter in zip(setters, splitters)]
    return splitters, setters, getters
