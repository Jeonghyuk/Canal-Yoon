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

# moved the imports inside the functions
# so you don't need to install all the dependencies
#import canal.image.codec.nifti
#import canal.image.codec.multitif
#import canal.image.codec.singletifs
import os

def autoloader(filename_or_s):
    """
    Selects a loader by checking the extension of the filename.

    Parameters
    ----------
    filename_or_s: str OR list of str

    Returns
    -------
    loader:
    """
    # supported extensions
    tif_ext = ('.tif', '.tiff')
    nifti_ext = ('.nifti',)
    ext_lists = (tif_ext, nifti_ext)

    if isinstance(filename_or_s, (tuple, list)): # multiple files (single tifs)
        filenames = filename_or_s # alias
        valid = all(os.path.splitext(name)[-1].lower() in tif_ext
                    for name in filenames) # whether all the filenames are of tif
        if valid:
            return SingleTifs.loader(filenames)
        else:
            raise ValueError('Multi loader is supported only for tifs. '
                             'Check the filenames (extensions).')
    else:   # single file
        filename = filename_or_s # alias
        ext = os.path.splitext(filename)[-1].lower() # the last one is extension
        if ext in tif_ext:
            return MultiTif.loader(filename)
        elif ext in nifti_ext:
            return Nifti.loader(filename)
        else:
            raise ValueError('''{} is not supported format.
                             Supported formats are:
                             {}'''.format(ext, [ext for ext_list in ext_lists
                                                for ext in ext_list]))
class SingleTifs:
    """
    Single tif I/O class. This class does NOT check the file extensions.
    """
    @staticmethod
    def loader(filenames):
        """
        Returns a loader for single tifs.

        Parameters
        ----------
        filenames: list of str

        Returns
        -------
        loader: SingleTifsLoader
        """
        import canal.image.codec.singletifs
        return canal.image.codec.singletifs.SingleTifsLoader(filenames)

class MultiTif:
    """
    Multi tif I/O class. This class does NOT check the file extension.
    """
    @staticmethod
    def loader(filename):
        """
        Returns a loader for multi-tif.

        Parameters
        ----------
        filename: str

        Returns
        -------
        loader: MultiTifLoader
        """
        import canal.image.codec.multitif
        return canal.image.codec.multitif.MultiTifLoader(filename)

class Nifti:
    """
    Nifti I/O class. This class does NOT check the file extension.
    """
    @staticmethod
    def loader(filename):
        """
        Returns a loader for nifti format file.

        Parameters
        ----------
        filename: str

        Returns
        -------
        loader: NiftiLoader
        """
        import canal.image.codec.nifti
        return canal.image.codec.nifti.NiftiLoader(filename)

    @staticmethod
    def dump(arr, filename):
        """
        Write ndarray into Nifti format.

        Parameters
        ----------
        arr: ndarray
        filename: str
        """
        import canal.image.codec.nifti
        canal.image.codec.nifti.dump(arr, filename)
