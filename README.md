Canal
=====

Python3 scripts for analyzing calcium imaging signals.

Registration (landmark based translation), cell detection, and parallel clustering (k-means) on the movies are implemented.

Prerequisites
-------------

numpy, scipy, scikit-learn, scikit-image, nibabel

ipython [optional]

Installation
------------

_On Ubuntu (tested under 16.04)_

sudo apt-get install python3 python3-numpy python3-scipy

sudo apt-get install python3-sklearn python3-skimage python3-nibabel

sudo apt-get install python3-nibabel

_On Windows (not tested)_

Download installers from the sites, and follow the instructions.

Attributes
----------

class _preprocessor.Movie_

Name | Description
-----|------------
New | loads movie from a file
to_nifti1 | saves movie as nifti 1 image
to_nifti2 | saves movie as nifti 2 image
get_data | gets the internal movie array (tzyx order)

class _preprocessor.MovieMapper_

Name | Description
---- | -----------
set\_landmarks\_properties | sets shape of the landmarks
set\_target\_time\_range | sets time range
get\_map | creates a _MovieMap_ class of the movie

class _preprocessor.MovieMap_

Name | Description
---- | -----------
get\_mapped | creates a mapped movie (_Movie_ class)

class _cluster.KMeans_

The interface is almost identical to scikit-learn _KMeans_ class.

Examples
--------

Registration on a movie.

> $ ipython3

> In [0]: from canal.preprocessor import Movie, MovieMap, MovieMapper	# import classes

> In [1]: movie = Movie.New('sample.nii')	# read nifti movie

> In [2]: mapper = MovieMapper()

> In [3]: mapper.set_landmark_properties((128, 128)) # landmark shape set to (128, 128)

> In [4]: mapper.set_target_time_range((0, 50)) # take median of the timeframes in 0:50

> In [5]: map = mapper.get_map(movie) # get MovieMap of the movie (take your time)

> In [6]: reg_movie = map.get_mapped(movie) # create movie applied the registration (take a coffee break)

Saving the movie (nifti 1 or 2 format).

> In [7]: affine = np.eye(4)

> In [8]: reg_movie.to_nifti1('reg_sample.nii', affine)

Cell detection.

> In [9]: from canal.cell import find_cell

> In [10]: data = reg_movie.get_data() # tzyx order numpy array

> In [11]: volume = data.mean(axis = 0) # make a 3d volume

> In [12]: cell_centers = find_cell(volume, 4) # find cells assuming their radii are 4