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
import itertools

class MultiTifLoader:
    """
    Loads a multi-tif format file.
    """
    def __init__(self, filename, verbose=True):
        """
        Parameters
        ----------
        filename: str
        verbose: bool, optional
        """
        self._filename = os.path.abspath(filename)
        self.verbose = verbose

    def load(self):
        """
        Loads whole image into the memory (ndarray).
        """
        buf = np.empty(self.shape, self.dtype)
        with open(self.filename, 'rb') as f:
            with PIL.Image.open(f) as i:
                if self.verbose:
                    print('Loading {}'.format(self._filename))
                with tqdm.trange(len(buf), unit='frame',
                                 disable=not self.verbose) as pbar:
                    for pos in pbar:
                        i.seek(pos)
                        buf[pos] = np.array(i)
        return buf

    @property
    def filename(self):
        return self._filename

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

    def _getitemfromlist(self, index):
        positions = index   # alias
        buf = np.empty((len(positions),) + self.shape[1:], self.dtype)
        with open(self.filename, 'rb') as f:
            with PIL.Image.open(f) as i:
                for num, pos in enumerate(positions):
                    i.seek(pos)
                    buf[num] = np.array(i)
        return buf

    def __getitem__(self, index):
        """
        Parameters
        ----------
        index: list or slice
        """
        if isinstance(index, list):
            return self._getitemfromlist(index)

        # extend index
        if isinstance(index, tuple):
            index = index + (slice(None),) * (self.ndim - len(index))
        else:
            index = (index,) + (slice(None),) * (self.ndim - 1)

        total = tuple(range(s) for s in self.shape)
        rindex = tuple(t[i] for t, i in zip(total, index))
        bshape = tuple(len(elem) for elem in rindex
                       if not isinstance(elem, int))

        tindex, vindex = rindex[0], index[1:]
        buf = np.empty(bshape, self.dtype)
        with open(self.filename, 'rb') as f:
            with PIL.Image.open(f) as i:
                try:
                    for num, pos in enumerate(tindex):
                        i.seek(pos)
                        buf[num] = np.array(i)[vindex]
                except TypeError:   # rindex[0] is not iterable
                    i.seek(tindex)
                    buf = np.array(i)[vindex]
        return buf

    @property
    def dtype(self):
        try:
            return self._dtype
        except AttributeError:
            with open(self.filename, 'rb') as f:
                with PIL.Image.open(f) as i:
                    self._dtype = np.array(i).dtype
            return self._dtype

    def _load_info(self):
        with open(self.filename, 'rb') as f:
            with PIL.Image.open(f) as i:
                first = np.array(i)
                planar_shape = first.shape
                self._dtype = first.dtype
                try:
                    if self.verbose:
                        print('Scanning {}'.format(self._filename))
                    with tqdm.tqdm(itertools.count(), unit='frame',
                                   disable=not self.verbose) as pbar:
                        for pos in pbar:
                            i.seek(pos)
                except EOFError:
                    self._shape = (pos,) + planar_shape
