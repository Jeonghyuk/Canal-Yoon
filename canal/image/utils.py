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

import itertools as itt
import numpy as np
import tqdm

class Splicer:
    """
    Splices loaders into a single image.
    """
    def __init__(self, loaders, ifrom=1, verbose=False):
        """
        Parameters
        ----------
        loaders: list
        ifrom: int, optional
            Immutable axes starts from `ifrom`.
        verbose: int, optional
            The number of immutable axes.
        """
        # checks validity
        first_loader = loaders[0]   # the first loader
        for loader in loaders:  # compare with the others
            if (loader.shape[ifrom:] != first_loader.shape[ifrom:] or   # shape
                loader.dtype != first_loader.dtype):                    # dtype
                raise ValueError('the number of dimensions must be identical')

        # set params
        self._ishape = first_loader.shape[ifrom:]   # immutable shape
        self._dtype = first_loader.dtype
        
        # set index map
        self._lindices = tuple((l, o) for l, loader in enumerate(loaders)
                               for o in range(len(loader)))
        self._mshape = (len(self._lindices),)       # mutable shape
        self._loaders = loaders
        self.verbose = verbose

    @property
    def dtype(self):
        return self._dtype

    def reshape(self, shape):
        ifrom = -len(self._ishape)
        mshape, ishape = shape[:ifrom], shape[ifrom:]
        if ishape != self._ishape:  # check ishape would not changed
            raise ValueError('cannot change the immutable shape')

        if np.product(mshape) != np.product(self._mshape):
            raise ValueError('cannot reshape array of size '
                             '{} into shape {}'.format(self.size, shape))
        else:
            self._mshape = mshape

    @property
    def shape(self):
        return self._mshape + self._ishape

    @shape.setter
    def shape(self, s):
        self.reshape(s)

    @property
    def size(self):
        return np.product(self.shape)

    @property
    def ndim(self):
        return len(self.shape)

    def __len__(self):
        return self.shape[0]
    '''
    def __getitem__(self, index):
        # extends index
        if isinstance(index, tuple):
            index = index + (slice(None),) * (self.ndim - len(index))
        else:
            index = (index,) + (slice(None),) * (self.ndim - 1)

        # resolves index
        total = [range(s) for s in self.shape]
        rindex = [t[i] for t, i in zip(total, index)]

        # prepares buffer
        bshape = tuple(len(r) for r in rindex if not isinstance(r, int))
        buf = np.empty(bshape, self.dtype)

        # creates points from the resolved index
        ifrom = -len(self._ishape)
        iindex = index[ifrom:]
        mpoints = list(itt.product(*[(i,) if isinstance(i, int) else i
                                     for i in rindex[:ifrom]]))
        bpoints = list(itt.product(*[range(len(i)) for i in rindex[:ifrom]
                                     if not isinstance(i, int)]))

        # loads image
        n_points = len(bpoints)
        if n_points > 1:    # the image consists of multiple slices
            with tqdm.tqdm(zip(bpoints, mpoints), unit='frame', total=n_points,
                           disable=not self.verbose) as pbar:
                for bpoint, mpoint in pbar:
                    imageslice = self._getslice(mpoint)
                    buf[bpoint] = imageslice[iindex]
        else:   # single slice
            mpoint = mpoints[0]
            imageslice = self._getslice(mpoint)
            buf = imageslice[iindex]
        return buf
    '''
    def __getitem__(self, index):
        # extends index
        if isinstance(index, tuple):
            index = index + (slice(None),) * (self.ndim - len(index))
        else:
            index = (index,) + (slice(None),) * (self.ndim - 1)

        # resolves index
        total = [range(s) for s in self.shape]
        rindex = [t[i] for t, i in zip(total, index)]

        # creates points from the resolved index
        ifrom = -len(self._ishape)
        iindex = index[ifrom:]
        mpoints = list(itt.product(*[(i,) if isinstance(i, int) else i
                                     for i in rindex[:ifrom]]))

        # prepares buffer
        bshape = tuple(len(r) for r in rindex if not isinstance(r, int))
        buf = np.empty((len(mpoints),) + self._ishape, self.dtype)

        # loads image
        moment = np.array(self._mshape[1:] + (1,))
        spoints = np.sum(mpoints * moment, axis=-1)
        lpoints = [self._lindices[spoint] for spoint in spoints]
        if len(lpoints) > 1:    # the image consists of multiple slices
            head = 0
            for key, group in itt.groupby(lpoints, key=lambda i: i[0]): # file
                offsets = [lp[-1] for lp in group]
                loader = self._loaders[key]
                n_offsets = len(offsets)
                buf[head:head + n_offsets] = loader[offsets][iindex]
                head += n_offsets
        else:   # single slice
            lpoint = lpoints[0]
            buf = loader[lpoint[0]][lpoint[-1]][iindex]
        return buf.reshape(bshape)
    '''
    def _getslice(self, mindex):
        moment = np.array(self._mshape[1:] + (1,))
        sindex = np.sum(mindex * moment)
        lindex = self._lindices[sindex]
        return self._loaders[lindex[0]][lindex[-1]]
    '''
