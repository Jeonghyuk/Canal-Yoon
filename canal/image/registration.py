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
import multiprocessing as mp
import itertools        # in lattice_points
import skimage.feature  # in find_features
import tqdm
import scipy.ndimage.filters
import scipy.optimize
import scipy.spatial.distance

def concatenate(*args):
    return Concatenator(args)

class Concatenator:
    """
    Concatenates ndarray of different shapes.
    """
    def __init__(self, args, ndim=None):
        if ndim is not None:
            for num, arg in enumerate(args):
                shape = (1,) * (ndim - arg.ndim) + arg.shape
                args[num] = arg.reshape(shape)

        self._data = args

    def _subindex(self, index):
        if not isinstance(index, int):
            raise ValueError('Index must be integer')

        lengths = [len(elem) for elem in self._data]
        ends = np.cumsum(lengths)
        for dindex, end in enumerate(ends):
            if index < end:
                return dindex, index - sum(lengths[:dindex])
        else:
            raise IndexError(('index {} is out of bounds for axis 0 ' + 
                              'with size {}').format(index, end))

    @property
    def ndim(self):
        return self._data[0].ndim

    @property
    def dtype(self):
        return self._data[0].dtype

    def __len__(self):
        return sum([len(elem) for elem in self._data])

    def __getitem__(self, index):
        dindex, sindex = self._subindex(index)
        return self._data[dindex][sindex]

class MovieMapper:
    """
    Translates movie using provided offsets. This class don't create the
    translated movie immediately. You should use `mapped` to create ndarray
    of translated movie.
    """
    def __init__(self, movie, offsets):
        """
        Creates a mapper of the movie.
        
        Parameters
        ----------
        movie: ndarray
        offsets: array-like of shape (T, D)
            The offsets of the features included in the movie. `D` should be
            less than or equal to ``movie.ndim - 1``. Also, `T` should be equal
            to ``len(movie)`` which is the number of time frames.
        """
        n_times = len(movie)
        movie_vdim = movie.ndim - 1 # number of dimensions of movie volume

        # checks the offsets
        offsets = np.asarray(offsets, int)
        offset_vdim = offsets.shape[-1] # number of dimensions of the offsets
        if offset_vdim < movie_vdim:    # extends offsets filling zeros
            extended = np.zeros((len(offsets), movie_vdim), int)
            extended[:, movie_vdim - offset_vdim:] = offsets
            offsets = extended
        elif offset_vdim > movie_vdim:
            raise ValueError('The condition offsets.shape[-1] <= movie.ndim - 1'
                             'is required')

        # prepares cropping
        # offset: position of feature in target relative to reference
        # canvas offset: offset of canvas which is needed to overlap features
        #                so `canvas_offset = -offset` would be a simple solution
        canvas_offsets = -offsets
        valid_begin = canvas_offsets.max(axis=0)
        valid_end = np.min(canvas_offsets + [elem.shape for elem in movie],
                           axis=0)
        cropped_shape = tuple(valid_end - valid_begin)
        crop_offsets = valid_begin - canvas_offsets

        self._movie = movie
        self._anchors = crop_offsets
        self._shape = (n_times,) + cropped_shape

    @property
    def shape(self):
        return self._shape

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def dtype(self):
        return self._movie.dtype

    def __len__(self):
        return self.shape[0]

    def __iter__(self):
        return (self._at(time) for time in range(len(self)))

    def _at(self, time):
        """
        Returns a movie frame at `time`.
        """
        if time < 0:
            time += len(self)
        anchor = self._anchors[time]
        vindex = tuple(slice(b, b + s) for b, s
                       in zip(anchor, self.shape[1:]))  # volume
        return self._movie[time][vindex]

    def mapped(self, verbose=False):
        """
        Returns the mapped movie.

        Parameters
        ----------
        verbose: bool

        Returns
        -------
        movie: ndarray
        """
        buf = np.empty(self.shape, self.dtype)
        if verbose:
            print('Creating movie')
        with tqdm.trange(len(self), unit='frame', disable=not verbose) as pbar:
            for bindex, time in enumerate(pbar):
                buf[bindex] = self._at(time)
        return buf

    def __getitem__(self, index):
        """
        Returns transformed movie.

        Parameters
        ----------
        index: slice
            A typical format of `__getitem__` function.

        Returns
        -------
        transformed: ndarray
        """
        # resolves index
        if isinstance(index, tuple):
            index = index + (slice(None),) * (self.ndim - len(index))
        else:
            index = (index,) + (slice(None),) * (self.ndim - 1)
        total = tuple(range(s) for s in self.shape)
        res_index = tuple(elem[i] for elem, i in zip(total, index)) # resolved

        # crops
        time_index = res_index[0]
        if isinstance(time_index, int): # directly call _at
            return self._at(time_index)[index[1:]]
        else:   # more than one time frame: creates buffer
            buf_shape = tuple(len(elem) for elem in res_index
                              if isinstance(elem, range))
            buf = np.empty(buf_shape, self.dtype)
            for bindex, time in enumerate(tindex):
                buf[bindex] = self._at(time)[index[1:]]
            return buf

def create_transform(kpseries, target_kps, kernel=1):
    """
    Calculate offsets from the keypoints.
    """
    offsets = [np.array([np.array(rkp) - tkp for rkp, tkp
                         in zip(kps, target_kps) if rkp is not None])
               for kps in kpseries]
    margin = 3 * kernel # 3 deviation
    min_offset = np.min([elem.min(axis=0) for elem in offsets], axis=0) - margin
    max_offset = np.max([elem.max(axis=0) for elem in offsets], axis=0) + margin

    # gets populations of the offsets
    edges = tuple(np.linspace(begin - 0.5, end + 0.5, end - begin + 2)
                  for begin, end in zip(min_offset, max_offset))
    hist = np.empty((len(offsets),) + tuple(len(elem) - 1 for elem in edges))
    for frame, frame_offsets in enumerate(offsets):
        hist[frame] = np.histogramdd(frame_offsets, edges)[0]
    hist = skimage.filters.gaussian(hist, kernel)

    # majority vote
    max_at = [np.unravel_index(frame_hist.argmax(), frame_hist.shape)
              for frame_hist in hist]
    centers = [range(begin, end) for begin, end
               in zip(min_offset, max_offset + 1)]
    return [tuple(c[i] for c, i in zip(centers, index)) for index in max_at]

# for multiprocessing
def _match_template(args):
    return skimage.feature.match_template(*args)

class Translation:
    """
    N-dimensional polynomial transform.
    """
    def __init__(self, polys):
        """
        Parameters
        ----------
        polys: list of np.poly1d
        """
        if all(isinstance(poly, np.poly1d) for poly in polys):
            self._polys = polys
        else:
            raise ValueError('Argument must be poly1d')

    def __call__(self, args):
        return tuple(poly(arg) for poly, arg in zip(self._polys, args))

    def __str__(self):
        format_str = "{} -> {}'"
        return '\n'.join([format_str.format(str(poly), poly.variable)
                          for poly in self._polys])

    def __repr__(self):
        return str(tuple(self._polys))

    def inverse(self):
        inverse_coeffs = []
        for poly in self._polys:
            c0, c1 = poly.coeffs
            inverse_coeff = [1 / c0, -c1 / c0]
            inverse_coeffs.append(inverse_coeff)
        return self.__class__(tuple(np.poly1d(c) for c in inverse_coeffs))

    def __setstate__(self, data):
        self._polys = data

    def __getstate__(self):
        return self._polys

class Embedder:
    def __init__(self, ambient_image):
        self._ambient_image = ambient_image

    def embed(self, subimage, interval, n_proc=None, verbose=False):
        if n_proc is None:
            n_proc = mp.cpu_count() - 1

        ambimage = self._ambient_image
        ## xy-plane registration
        # poorman's registration
        substack = subimage.max(axis=0)
        detector = FeatureDetector((256, 256), (16, 16), 32, 256)   # fix
        kps, features = detector.detect(substack, n_proc=n_proc,
                                        verbose=verbose)
        tracker = FeatureTracker(kps, features)
        ambstacks = np.array([ambimage[b::interval].max(axis=0)
                              for b in range(interval)])
        kpseries = tracker.track(ambstacks, 8, n_proc=n_proc, verbose=verbose)
        amboffset = np.median(create_transform(kpseries, kps), axis=0)
        if verbose:
            print('XY shift detected: {}'.format(amboffset))

        # map
        offsets = [(0, 0)] * len(subimage) + [amboffset] * len(ambimage)
        concatenated = concatenate(subimage, ambimage)
        planar_mapper = MovieMapper(concatenated, offsets)
        mapped = planar_mapper.mapped(verbose=verbose)
        subimage, ambimage = mapped[:len(subimage)], mapped[len(subimage):]

        ## z axis embedding
        # evaluate distances
        distances = np.empty((len(subimage), len(ambimage)))
        if verbose:
            print('Measuring distances')
        with tqdm.tqdm(itertools.product(*[range(s) for s in distances.shape]),
                       total=distances.size, unit='pair',
                       disable=not verbose) as pbar:
            for num, (subindex, ambindex) in enumerate(pbar):
                u, v = subimage[subindex].flat, ambimage[ambindex].flat
                distances.flat[num] = scipy.spatial.distance.euclidean(u, v)

        # normalize distances
        amb_moment = np.sqrt([np.sum(elem * elem) for elem in ambimage])
        distances /= amb_moment
        sub_moment = np.sqrt([np.sum(elem * elem) for elem in subimage])
        distances /= sub_moment[:, np.newaxis]

        # cost function
        def cost(a0, a1=interval, distances=distances):   # a1 * x + a0
            fit = np.poly1d([a1, a0])
            n_sub, n_amb = distances.shape
            subindices = range(n_sub)
            ambindices = [int(round(elem)) for elem in fit(subindices)]
            try:
                caught = [distances[index] for index in zip(subindices,
                                                            ambindices)]
                return np.mean(caught)
            except IndexError:
                return np.nan

        # brute force search
        grid = (slice(0, int(len(ambimage) - interval * len(subimage)), 0.1),)
        result = scipy.optimize.brute(cost, grid,
                                      finish=scipy.optimize.fmin)
        return Translation([np.poly1d([interval, result], variable='z'),
                            np.poly1d([1, amboffset[0]], variable='y'),
                            np.poly1d([1, amboffset[1]], variable='x')])

class FeatureTracker:
    """
    Tracks the feature in the movie.
    """
    def __init__(self, keypoints, features):
        """
        Parameters
        ----------
        keypoints: tuple of length N
        features: tuple of length N
        """
        self._keypoints = keypoints
        self._features = features

    def track(self, movie, tolerance, n_proc=None, verbose=False):
        """
        Tracks the features.

        Paramters
        ---------
        movie: ndarray
        tolerance: int
        n_proc: int, optional
        verbose: bool, optional

        Returns
        -------
        keypoint_series: list
        """
        if n_proc is None:  # default parameter for n_proc
            n_proc = mp.cpu_count() - 1

        features = self._features
        target_keypoints = self._keypoints
        n_sdim = movie.ndim - 1 # spatial dimension
        n_times = len(movie)
        search_indices = [tuple(slice(0, None) for d in range(n_sdim))
                          for num in range(len(features))]

        def guess_indices(offset, image_shape=movie.shape[1:],
                          feature_shapes=[f.shape for f in features],
                          target_keypoints=target_keypoints, tolerance=tolerance):
            validity, indices = [], []
            for keypoint, shape in zip(target_keypoints, feature_shapes):
                point = tuple(p + o for p, o in zip(keypoint, offset))
                # cache for next search
                begin = [p - tolerance for p in point]
                end = [p + tolerance + 1 + s for s, p in zip(shape, point)]
                # check index range
                if (all([b >= 0 for b in begin]) and
                    all([e <= s for e, s in zip(end, image_shape)])):
                    validity.append(True)
                else:
                    validity.append(False)  # outside of image
                index = tuple(slice(b, e) for b, e in zip(begin, end))
                indices.append(index)
            return validity, indices

        def fill_mask(mask, fill, constant):
            dispenser = iter(fill)
            return tuple(next(dispenser) if flag else constant
                         for flag in mask)

        with mp.Pool(processes=n_proc) as pool:
            # initial search
            args = ((movie[0], feature) for feature in features)
            if verbose:
                print('Initial search')
            with tqdm.tqdm(args, total=len(features), unit='feature',
                           disable=not verbose) as pbar:
                keypoints = [elem for elem in pool.imap(_locate_feature, pbar)]
            offset = match_points(keypoints, target_keypoints)
            validity, next_indices = guess_indices(offset)
                
            # tracking
            offsets = np.empty((n_times, n_sdim), int)
            tracked = []
            if verbose:
                print('Tracking')
            with tqdm.trange(n_times, unit='frame', disable=not verbose) as pbar:
                for time in pbar:
                    args = ((movie[time][index], feature, index, False)
                            for index, feature, valid in zip(next_indices,
                                                             features, validity)
                            if valid)
                    keypoints = pool.map(_locate_feature, args)
                    valid_points = [e for e, valid
                                    in zip(target_keypoints, validity) if valid]
                    offset = match_points(keypoints, valid_points)
                    offsets[time] = offset
                    tracked.append(fill_mask(validity, keypoints, None))
                    validity, next_indices = guess_indices(offset)
            return tracked

def lattice_points(volume_shape, interval, margin):
    """
    Returns lattice points.

    Parameters
    ----------
    volume_shape: tuple of length D
        Shape of D-dimensional lattice.
    interval: tuple of length D
    margin: tuple of length D

    Returns
    -------
    points: list
        lattice points in D-dimensional space.
    """
    ranges = [range(m, s - m + 1, i)
              for i, m, s in zip(interval, margin, volume_shape)]
    points = list(itertools.product(*ranges))
    return points

def _feature_quality(args):
    img, feature, answer = args
    imgblur = skimage.filters.gaussian(img.astype(float), 1)
    featureblur = skimage.filters.gaussian(feature.astype(float), 1)
    imgnoise = np.std(img - imgblur)
    imgspice = imgnoise * (np.random.rand(*img.shape) * 2 - 1)
    score = skimage.feature.match_template(imgblur + imgspice, featureblur)
    #return scipy.stats.kurtosis(score, axis=None)
    if score.max() > score[answer]:
        return -np.inf
    else:
        return (score[answer] - np.median(score)) / score.std()

class FeatureDetector:
    def __init__(self, shape, stride, min_distance=None, n_keypoints=None):
        """
        Finds keypoint from the image, whose pattern is a lattice determined
        by `stride`.

        Parameters
        ----------
        shape: tuple of length D
            A shape of the features.
        stride: tuple of length D
            A stride of the lattice.
        min_distance: float, optional
            The minimum distances between the keypoints.
        n_keypoints: int, optional
            If not ``None``, the best `n_keypoints` keypoints would be selected.
        """
        self._shape = shape
        self._stride = stride
        self._min_distance = min_distance
        self._n_keypoints = n_keypoints

    def detect(self, image, n_proc=None, verbose=False):
        """
        Detects keypoints and features from the image.

        Parameters
        ----------
        image: ndarray
            An image from which the detector finds features.
        n_proc: int, optional
            The number of processes to use. If set to default (``None``), 
            all cpu would be used.
        verbose: bool, optional
            If ``True``, the progress would be printed.

        Returns
        -------
        keypoints: tuple
        features: tuple
        """
        if n_proc is None:  # default parameter for n_proc
            n_proc = mp.cpu_count() - 1

        shape = self._shape
        stride = self._stride
        min_distance = self._min_distance
        n_keypoints = self._n_keypoints

        # sampling (lattice pattern)
        keypoints = lattice_points(image.shape, stride, shape)
        
        # feature index
        feature_indices = [tuple(slice(origin, origin + width)
                                 for origin, width in zip(point, shape))
                           for point in keypoints]
        features = tuple(image[index] for index in feature_indices)
        
        # evaluation index (local window for accuracy evaluation)
        eval_indices = [tuple(slice(i.start - width, i.stop + width)
                                for i, width in zip(index, shape))
                          for index in feature_indices]

        # evaluates the performances of the features
        if n_keypoints is not None or min_distance is not None:
            with mp.Pool(processes=n_proc) as pool:
                args = ((image[index], feature, shape)
                        for index, feature in zip(eval_indices, features))
                #args = ((image, feature) for feature in features)
                scores = np.empty(len(features))
                if verbose:
                    print('Inspecting features')
                with tqdm.tqdm(args, total=len(features), unit='feature',
                               disable=not verbose) as pbar:
                    for f_num, quality in enumerate(pool.imap(_feature_quality,
                                                              pbar)):
                        scores[f_num] = quality

            # sort
            ranked_ids = sorted(range(len(scores)), key=lambda i: scores[i],
                                reverse=True)
            keypoints = [keypoints[i] for i in ranked_ids]
            features = [features[i] for i in ranked_ids]

        # check min distance
        if min_distance is not None:
            if verbose:
                print('Checking distances between the key points')
            with tqdm.trange(len(keypoints), unit='point',
                             disable=not verbose) as pbar:
                selected_ids = []
                for cid in pbar:
                    cpt = keypoints[cid]    # candidate point
                    for sid in selected_ids:
                        spt = keypoints[sid]
                        distance = scipy.spatial.distance.euclidean(cpt, spt)
                        if distance < min_distance:
                            break
                    else:
                        selected_ids.append(cid)
                    if (n_keypoints is not None and
                        len(selected_ids) == n_keypoints):
                        break
            keypoints = [keypoints[i] for i in selected_ids]
            features = [features[i] for i in selected_ids]

        # select keypoints and features
        if n_keypoints is not None:
            if verbose and len(keypoints) < n_keypoints:
                print('{} (< {}) key points found'.format(len(keypoints),
                                                          n_keypoints))
            keypoints = tuple(keypoints[:n_keypoints])
            features = tuple(features[:n_keypoints])
        return keypoints, features

def match_points(reference_points, target_points):
    """
    Returns most frequent offset from target to reference.

    Parameters
    ----------
    reference_points: list of length N (equivalent of ndarray of shape (N, D))
        N points in D-dimensional space. All elements should be tuple of size D.
    target_points: list of length N (equivalent of ndarray of shape (N, D))
        N points in D-dimensional space. All elements should be tuple of size D.

    Returns
    -------
    offset: tuple of length D
    """
    offsets = np.asarray(reference_points) - np.asarray(target_points)
    #mean_offset, std_offset = offsets.mean(axis=0), offsets.std(axis=0)
    hist_range = tuple(zip(offsets.min(axis=0), offsets.max(axis=0)))
    hist_edges = [np.linspace(begin - 0.5, end + 0.5, end - begin + 2)
                  for begin, end in hist_range]
    hist = skimage.filters.gaussian(np.histogramdd(offsets, hist_edges)[0], 1)
    max_at = np.unravel_index(hist.argmax(), hist.shape)
    left_edge = [edges[index] for edges, index in zip(hist_edges, max_at)]
    return tuple(np.ceil(left_edge).astype(int))

# for multiprocessing
def _locate_feature(args):
    return locate_feature(*args)

def locate_feature(image, feature, index=None, apply_mask=True):
    """
    Returns position of feature in image.

    Parameters
    ----------
    image: ndarray of shape (A, B[, C])
    feature: ndarray of shape (a, b[, c])
    index: tuple, optional
        A mask in which it searchs for feature.
    apply_mask: bool, optional
        If True (by default), `index` would be applyed to image before search.
        If False, the offset extracted from the index would be applied to the
        output, however, `index` would NOT affect the search.
        Implemented for the performance reason.

    Returns
    -------
    point: tuple of length 2 (or 3)
    """
    search = image[index] if index is not None and apply_mask else image
    score = skimage.feature.match_template(search, feature)
    max_at = np.unravel_index(score.argmax(), score.shape)

    if index is not None:   # offset
        extended_index = index + (slice(0, None),) * (image.ndim - len(index))
        offset = np.array([i.start for i in extended_index])
        max_at = tuple(max_at + offset)
    return max_at
