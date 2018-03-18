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
import PIL.Image
import os
import tqdm

class SingleTifsLoader:
    """
    Loads a series of single tif format files.
    """
    def __init__(self, filenames, verbose=True):
        """
        Parameters
        ----------
        filenames: list of str
        verbose: bool, optional
        """
        self._filenames = tuple([os.path.abspath(elem) for elem in filenames])
        self.verbose = verbose

    def load(self):
        """
        Loads whole image into the memory (ndarray).
        """
        buf = np.empty(self.shape, self.dtype)
        if self.verbose:
            print('Loading files')
        with tqdm.tqdm(self.filenames, unit='frame',
                       disable=not self.verbose) as pbar:
            for pos, filename in enumerate(pbar):
                with open(filename, 'rb') as f:
                    with PIL.Image.open(f) as i:
                        buf[pos] = np.array(i)
        return buf

    @property
    def filenames(self):
        return self._filenames

    @property
    def shape(self):
        try:
            return self._shape
        except AttributeError:
            self._load_info()
            return self._shape

    def __len__(self):
        return self.shape[0]

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def size(self):
        return np.product(self.shape)

    def __getitem__(self, index):
        # extend index
        if isinstance(index, tuple):
            index = index + (slice(None),) * (self.ndim - len(index))
        else:
            index = (index,) + (slice(None),) * (self.ndim - 1)

        total = ((self.filenames[index[0]],) +
                 tuple(range(s) for s in self.shape[1:]))
        rindex = tuple(elem[i] for elem, i in zip(total, index)) # resolve
        bshape = tuple(len(elem) for elem in rindex
                       if not isinstance(elem, int))

        tindex, vindex = rindex[0], index[1:]
        buf = np.empty(buf_shape, self.dtype)
        try:
            for num, filename in enumerate(tindex):
                with open(filename, 'rb') as f:
                    with PIL.Image.open(f) as i:
                        buf[num] = np.array(i)[vindex]
        except TypeError:   # res_index[0] is not iterable
            with open(res_index[0], 'rb') as f:
                with PIL.Image.open(f) as i:
                    buf = np.array(i)[vindex]
        return buf

    @property
    def dtype(self):
        try:
            return self._dtype
        except AttributeError:
            self._load_info()
            return self._dtype

    def _load_info(self):
        filename = self.filenames[0]
        with open(filename, 'rb') as f:
            with PIL.Image.open(f) as i:
                first = np.array(i)
        self._dtype = first.dtype
        self._shape = (len(self.filenames),) + first.shape
