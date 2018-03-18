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
import nibabel as nib
import os
import tqdm

def dump(arr, filename):
    """
    Dumps the array into nifti file. It checks the array could be stored as
    Nifti1 format. If not, stores the array as Nifti2 format.

    Parameters
    ----------
    arr: ndarray
    filename: str
    """
    permutation = tuple(range(arr.ndim)[::-1])
    if np.all(np.array(arr.shape) < 65536 // 2):    # OK to go with Nifti1
        i = nib.Nifti1Image(arr.transpose(permutation), np.eye(4))
    else:   # only with Nifti2
        i = nib.Nifti2Image(arr.transpose(permutation), np.eye(4))
    nib.save(i, filename)

class NiftiLoader:
    def __init__(self, filename):
        """
        
        Parameters
        ----------
        filename: string
            A file name of the nifti image.
        """
        self._filename = os.path.abspath(filename)

    def load(self):
        """
        Returns
        -------
        arr: ndarray or memmap
        """
        arr = nib.load(self.filename).get_data('unchanged')
        permutation = tuple(range(arr.ndim)[::-1])
        return arr.transpose(permutation)

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

    def __getitem__(self, index):
        return self.load()[index]

    @property
    def dtype(self):
        try:
            return self._dtype
        except AttributeError:
            self._load_info()
            return self._dtype

    def _load_info(self):
        i = nib.load(self.filename)
        self._dtype = i.get_data_dtype()
        self._shape = i.shape[::-1]
