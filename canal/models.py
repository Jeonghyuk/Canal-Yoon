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
import scipy.ndimage.filters
import multiprocessing as mp
import itertools
import mpmath as ap
import wrappers.mpmath as hp
import random
import tqdm

class Quantizer:
    def __init__(self, n_levels):
        self._n_levels = n_levels
    
    def quantize(self, raw):
        return np.array([self._quantize(elem) for elem in raw])
    
    def _quantize(self, raw):
        std = raw.std()
        max_std = self._n_levels - 1
        digital = np.ones(len(raw), int) * max_std
        for thres in range(max_std)[::-1]:
            digital[raw < std * (thres + 1)] = thres
        return digital
    
class SpatialCodec:
    def __init__(self, observation):
        self._n_pos = len(observation)
        self._radix = observation.max() + 1
    
    def encode(self, raw):
        moment = self._radix ** np.arange(self._n_pos)
        return np.dot(moment, raw)

class TemporalCodec:
    def __init__(self, observation):
        id_size = observation.max() + 1
        valid_ids = np.array([np.any(observation == oid)
                              for oid in range(id_size)])

        decoder = np.arange(id_size)[valid_ids]
        encoder = np.ones(id_size, int) * -1
        for encoded_id, raw_id in enumerate(decoder):
            encoder[raw_id] = encoded_id
        
        self._encoder = encoder
        self._decoder = decoder
    
    def encode(self, raw_id):
        if raw_id < len(self._encoder):
            cand_id = self._encoder[raw_id]
            if cand_id != -1:
                return cand_id
        raise ValueError('Invalid raw id')
            
    def decode(self, encoded_id):
        if encoded_id < len(self._decoder):
            return self._decoder[encoded_id]
        raise ValueError('Invalid encoded id')
        
    def encode_observation(self, raw):
        return np.array([self.encode(elem) for elem in raw])
    
    def encode_emission(self, raw):
        removed = raw.T[self._decoder].T
        return removed / removed.sum(axis = -1, keepdims = True)

class Simulator:
    def __init__(self, transition, emission):
        self._transition = transition
        self._emission = emission
    
    def simulate_states(self, n_times, init_dist = None):
        states = np.empty(n_times, int)
        transition = self._transition
        
        # initial state
        random.seed()
        if init_dist is None:
            rec = transition.diagonal()
            norm_rec = rec / rec.sum()
            states[0] = Simulator._draw(norm_rec)
        elif isinstance(init_dist, int):
            states[0] = init_dist
        elif iterable(init_dist):
            arrdist = np.array(init_dist)
            norm_dist = arrdist / arrdist.sum()
            states[0] = Simulator._draw(norm_dist)
        else:
            raise ValueError('initial distribution must be None or integer or '
                             '1-d iterable')
        
        # get sequence of states
        for time in range(1, n_times):
            states[time] = Simulator._draw(transition[states[time - 1]])
        
        return states
    
    def simulate_observations(self, states):
        emission = self._emission
        random.seed()
        return np.array([Simulator._draw(emission[state]) for state in states])
    
    def _draw(probabilities):
        rand = random.random()
        for index, val in enumerate(np.cumsum(probabilities)):
            if rand < val:
                return index

def state_masks(state_begin, state_end, path):
    masks = []
    begun = False
    for time, state in enumerate(path):
        if state == state_begin:
            begun = True
            time_begin = time
        elif begun and state == state_end:
            begun = False
            masks.append(slice(time_begin, time))
    return masks

def _shuffledot(args):
    raw, shuf = args
    np.random.shuffle(shuf)
    return np.dot(raw, shuf)

def _shuffledotmulti(args):
    raw, shuf, n_shuf = args
    ret = np.empty((n_shuf, len(raw), len(shuf.T)))
    np.random.seed()
    for it in range(n_shuf):
        np.random.shuffle(shuf)
        ret[it] = np.dot(raw, shuf)
    return ret

def binary_profile(signals, indices):
    # range of mask: (-1, 1)
    n_times = signals.shape[-1]
    mask = np.zeros((len(indices), n_times), bool)
    for num, index in enumerate(indices):
        mask[num][index] = 1
    return np.dot(signals, mask.T).astype(bool)

def binary_profile_test(signals, positive, negative,
                        n_iter=512, n_proc=None, verbose=False):
    if n_proc is None:
        n_proc = mp.cpu_count() - 1
    # range of mask: (-1, 1)
    n_times = signals.shape[-1]
    mask = positive.astype(int) - negative.astype(int)
    mask = mask.reshape(1, mask.size)

    if n_proc is None:
        n_proc = mp.cpu_count() - 1

    # range of mask: (-1, 1)
    momented = np.dot(signals, mask.T)
    shuffle_momented = np.empty((n_iter,) + momented.shape)
    if verbose:
        print('Creating shuffled distribution')

    if verbose:
        with tqdm.trange(n_iter) as pbar:
            with mp.Pool(processes=n_proc) as pool:
                args = ((signals, mask.T) for i in pbar)
                for it, ret in enumerate(pool.imap(_shuffledot, args)):
                    shuffle_momented[it] = ret
    else:
        with mp.Pool(processes=n_proc) as pool:
            workiters = np.linspace(0, n_iter, n_proc + 1).astype(int)
            worknums = workiters[1:] - workiters[:-1]
            args = ((signals, mask.T, num) for num in worknums)
            shuffle_momented = np.concatenate(pool.map(_shuffledotmulti, args))

    #return (momented - momented.mean(axis=0)) / momented.std(axis=0)
    return momented
    
def profile(signals, path, n_iter=512, n_proc=None, verbose=False):
    """
    Parameters
    ----------
    signals: ndarray
    path: ndarray
    states: slice
    n_iter: int
    n_proc: int
    verbose: bool
    """
    if n_proc is None:
        n_proc = mp.cpu_count() - 1

    # range of mask: (-1, 1)
    n_times = signals.shape[-1]
    n_states = path.max() + 1
    mask = np.ones((n_states, n_times)) * 0 #-1
    for state in range(n_states):
        mask[state][state == path] = 1

    momented = np.dot(signals, mask.T)
    shuffle_momented = np.empty((n_iter,) + momented.shape)
    if verbose:
        print('Creating shuffled distribution')

    if verbose:
        with tqdm.trange(n_iter) as pbar:
            with mp.Pool(processes=n_proc) as pool:
                args = ((signals, mask.T) for i in pbar)
                for it, ret in enumerate(pool.imap(_shuffledot, args)):
                    shuffle_momented[it] = ret
    else:
        with mp.Pool(processes=n_proc) as pool:
            workiters = np.linspace(0, n_iter, n_proc + 1).astype(int)
            worknums = workiters[1:] - workiters[:-1]
            args = ((signals, mask.T, num) for num in worknums)
            shuffle_momented = np.concatenate(pool.map(_shuffledotmulti, args))

    return (momented - momented.mean(axis=0)) / momented.std(axis=0)

def direct_profile(signals, path, verbose=False):
    """
    Parameters
    ----------
    signals: ndarray
    path: ndarray
    states: slice
    n_iter: int
    n_proc: int
    verbose: bool
    """

    # range of mask: (-1, 1)
    n_times = signals.shape[-1]
    n_states = path.max() + 1
    mask = np.ones((n_states, n_times)) * 0
    for state in range(n_states):
        mask[state][state == path] = 1
    mask /= mask.sum(axis=-1, keepdims=True)

    momented = np.dot(signals, mask.T)
    return momented

def out_scatter(locations, masks, titles):
    import matplotlib.pyplot as plt
    import matplotlib.gridspec
    z, y, x = locations.T
    gs = matplotlib.gridspec.GridSpec(2, 1, hspace=0)
    for title, mask in zip(titles, masks):
        fig = plt.figure()
        ax = plt.subplot(gs[0])
        if mask.dtype == bool:
            ax.scatter(x, y, color='grey')
            ax.scatter(x[mask], y[mask], color='red')
        else:
            ax.scatter(x, y, c=mask)
        plt.ylabel('Y')
        plt.title(title)
        ax = plt.subplot(gs[1], sharex=ax)
        if mask.dtype == bool:
            ax.scatter(x, z, color='grey')
            ax.scatter(x[mask], z[mask], color='red')
        else:
            ax.scatter(x, z, c=mask)
        plt.ylabel('Z')
        plt.xlabel('X')
        plt.xlim(0, 1024)
        plt.show()
        
def out_plot(signals, masks, titles):
    import matplotlib.pyplot as plt
    import matplotlib.gridspec
    gs = matplotlib.gridspec.GridSpec(2, 1, hspace=0, height_ratios=[10, 1])
    for title, mask in zip(titles, masks):
        fig = plt.figure()
        ax = plt.subplot(gs[0])
        ax.plot(signals[mask].T)
        plt.ylabel(r'$\Delta F / F$')
        plt.title(title)
        ax = plt.subplot(gs[1], sharex=ax)
        ax.imshow(mask.reshape((1, mask.size)), interpolation='none', aspect='auto')
        plt.xlabel('Time [frame]')
        plt.xlim(0, signals.shape[-1])
        plt.show()

def state_labels(prefix, n_phases, quiescence=True):
    n_digit = len(str(n_phases - 1))
    format_str = '{}{{:0{}}}'.format(prefix, n_digit)
    labels = [format_str.format(num) for num in range(n_phases)]
    if quiescence:
        labels = [prefix + 'Q'] + labels
    return labels

def canonical_labels(n_forward, n_backward, n_turning):
    return (state_labels('S', 0) + state_labels('F', n_forward) +
            state_labels('B', n_backward) + state_labels('TL', n_turning) +
            state_labels('TR', n_turning))

def transition_matrix(n_forward, n_backward, n_turning,
                      propagate, trigger, cool, excite):
    # high precision
    propagate = ap.mpf(str(propagate))
    trigger = ap.mpf(str(trigger))
    cool = ap.mpf(str(cool))
    excite = ap.mpf(str(excite))
    
    def subtransition(n_states, trigger, cool, propagate):
        sub = hp.diag([1.0 - trigger - cool] + [1.0 - propagate] * n_states)
        q_id = 0
        sub[q_id, q_id + 1] = trigger
        for row in range(0, -n_states, -1):
            sub[row - 1, row] = propagate
        return sub
    
    prob_arr = hp.zeros((n_forward + 1 +
                         n_backward + 1 +
                         (n_turning + 1) * 2 +
                         1,) * 2)
    sq_id = 0                           # id of steady quiscence
    eqf_id = sq_id + 1                  # id of excited quiescence for forward wave
    eqb_id = eqf_id + n_forward + 1     # id of excited quiescence for backward wave
    eqtl_id = eqb_id + n_backward + 1   # id of excited quiescence for left turning
    eqtr_id = eqtl_id + n_turning + 1   # id of excited quiescence for right turning
    
    # for steady quiescence
    prob_arr[sq_id, sq_id] = 1 - 4 * excite
    prob_arr[sq_id, eqf_id] = excite
    prob_arr[sq_id, eqb_id] = excite
    prob_arr[sq_id, eqtl_id] = excite
    prob_arr[sq_id, eqtr_id] = excite
    
    # for excited quiescence which triggers forward wave
    prob_arr[eqf_id, sq_id] = cool
    
    # for excited quiescence which triggers backward wave
    prob_arr[eqb_id, sq_id] = cool
    
    # for excited quiescence which triggers turning
    prob_arr[eqtl_id, sq_id] = cool
    prob_arr[eqtr_id, sq_id] = cool
    
    # for forward wave
    forward_arr = subtransition(n_forward, trigger, cool, propagate)
    prob_arr[eqf_id:eqb_id, eqf_id:eqb_id] = forward_arr
    
    # for backward wave
    backward_arr = subtransition(n_backward, trigger, cool, propagate)
    prob_arr[eqb_id:eqtl_id, eqb_id:eqtl_id] = backward_arr
    
    # for left turning
    turning_arr = subtransition(n_turning, trigger, cool, propagate)
    prob_arr[eqtl_id:eqtr_id, eqtl_id:eqtr_id] = turning_arr
    
    # for right turning
    prob_arr[eqtr_id:, eqtr_id:] = turning_arr
    
    return prob_arr / prob_arr.sum(axis = -1, keepdims = True)
    
def emission_matrix(n_waves, n_turnings, sigma):
    n_states = (n_waves + 1) * 2 + (n_turnings + 1) * 2 + 1
    n_cells = n_waves * 2
    n_obs = 2 ** n_cells
    emit_arr = hp.zeros((n_states, n_obs))
    
    # quiscence
    silent = ap.mpf(n_waves)
    spon = ap.mpf(1)
    
    sq_id = 0
    emit_arr[sq_id] = spon
    emit_arr[sq_id, 0] = silent
    
    eqf_id = sq_id + 1
    emit_arr[eqf_id] = spon
    emit_arr[eqf_id, 0] = silent
    
    eqb_id = eqf_id + n_waves + 1
    emit_arr[eqb_id] = spon
    emit_arr[eqb_id, 0] = silent
    
    eqtl_id = eqb_id + n_waves + 1
    emit_arr[eqtl_id] = spon
    emit_arr[eqtl_id, 0] = silent
    
    eqtr_id = eqtl_id + n_turnings + 1
    emit_arr[eqtr_id] = spon
    emit_arr[eqtr_id, 0] = silent
    
    # seed for templates
    seed = np.eye(n_waves)
    blur = np.array([scipy.ndimage.filters.gaussian_filter1d(vec, sigma)
                     for vec in seed])
    
    def subemission(template, n_states):
        n_cells = len(template)
        emission = hp.zeros((n_states, 2 ** n_cells))
        for o_id, mask in enumerate(itertools.product(
                                    *itertools.repeat((False, True), n_cells))):
            for s_id in range(n_states):
                selected = template[np.array(mask[::-1]), s_id]
                emission[s_id][o_id] = selected.mean() if len(selected) != 0 else 0
        return emission
        
    # forward wave
    forward_template = hp.asmparray(np.r_[blur, blur][:, ::-1])
    emit_arr[eqf_id + 1:eqb_id] = subemission(forward_template, n_waves)
    
    # backward wave
    backward_template = forward_template[:, ::-1]
    emit_arr[eqb_id + 1:eqtl_id] = subemission(backward_template, n_waves)

    #anterior_subtemplate = np.zeros_like(blur)
    #anterior_subtemplate[:n_turnings] = blur[-n_turnings:, ::-1]

    # left turning
    #left_turning_template = hp.asmparray(np.r_[anterior_subtemplate,
    #                                           np.zeros(blur.shape)])
    #emit_arr[eqtl_id + 1:eqtr_id] = subemission(left_turning_template, n_turnings)
    
    # right turning
    #right_turning_template = hp.asmparray(np.r_[np.zeros(blur.shape),
    #                                            anterior_subtemplate])
    #emit_arr[eqtr_id + 1:] = subemission(right_turning_template, n_turnings)
    # left turning
    left_turning_template = hp.asmparray(np.r_[blur, np.zeros(blur.shape)])
    emit_arr[eqtl_id + 1:eqtr_id] = subemission(left_turning_template, n_turnings)
    
    # right turning
    right_turning_template = hp.asmparray(np.r_[np.zeros(blur.shape), blur])
    emit_arr[eqtr_id + 1:] = subemission(right_turning_template, n_turnings)
    
    return emit_arr / emit_arr.sum(axis = -1, keepdims = True)

def smooth(vec, noise_sigma, background_sigma):
    blur = scipy.ndimage.filters.gaussian_filter1d(vec, noise_sigma)
    background = scipy.ndimage.filters.gaussian_filter1d(vec, background_sigma)
    return blur / background - 1
    
def viterbi_path(observation, transition, emission, init_dist, verbose=False):
    '''
    returns viterbi path.
    
    Parameters
    ----------
    observation: integer array of shape (T,)
    observation[i] is the observation on time i.
    
    transition: float array of shape (S, S)
    transition[i, j] is the transition probability of transiting from state i to state j.
    
    emission: float array of shape (S, O)
    emission[i, j] is the probability of observing j from state i.
    
    init_dist: float array of shape (S,)
    init_dist[i] is the initial probability of being state i.
    
    Returns
    -------
    viterbi_path: integer array of shape (T,)
    The most likely state sequence.
    '''
    
    # input parameter validation
    if observation.dtype != int:
        raise ValueError('Condition observation.dtype == int not met.')
    
    n_states, n_obs = emission.shape
    if transition.shape != (n_states,) * 2:
        raise ValueError('')
    if init_dist.shape != (n_states,):
        raise ValueError('')
    
    if not observation.max() < n_obs:
        raise ValueError('')
    if not hp.array_equiv(hp.asmparray(transition, ap.iv.mpf).sum(axis=-1), 1):
        raise ValueError('')
    if not hp.array_equiv(hp.asmparray(emission, ap.iv.mpf).sum(axis=-1), 1):
        raise ValueError('')
    if not hp.array_equiv(hp.asmparray(init_dist, ap.iv.mpf).sum(), 1):
        raise ValueError('')
    
    # most likely path (viterbi path)
    n_times = len(observation)
    path_probs = hp.zeros((n_times, n_states))
    path_states = np.empty((n_times, n_states), int)

    if verbose:
        print('Determining state for each frame')
    path_probs[0] = emission[:, observation[0]] * init_dist
    with tqdm.trange(1, n_times, unit='frame', disable=not verbose) as pbar:
        for time in pbar:
            for state in range(n_states):
                every_probs = (path_probs[time - 1] * transition[:, state] *
                               emission[state, observation[time]])
                max_prob_state = every_probs.argmax()
                
                path_states[time, state] = max_prob_state
                path_probs[time, state] = every_probs[max_prob_state]
    
    # viterbi path
    viterbi_path = np.empty(n_times, int)
    viterbi_path[-1] = path_probs[-1].argmax()
    for time in range(1, n_times)[::-1]:
        viterbi_path[time - 1] = path_states[time, viterbi_path[time]]
        
    return viterbi_path

def baum_welch_iter(observation, transition, emission, init_dist,
                    n_iter, verbose=False):
    for num in range(n_iter):
        if verbose:
            print('Iter {}'.format(num))
        transition, emission, init_dist = baum_welch(observation, transition,
                                                     emission, init_dist,
                                                     verbose=verbose)
    return transition, emission, init_dist

def baum_welch_converge(observation, transition, emission, init_dist,
                        verbose=False):
    prev_viterbi = viterbi_path(observation, transition, emission, init_dist,
                                verbose)
    for num in itertools.count():
        if verbose:
            print('Iter {}'.format(num))
        transition, emission, init_dist = baum_welch(observation, transition,
                                                     emission, init_dist,
                                                     verbose=verbose)
        next_viterbi = viterbi_path(observation, transition, emission,
                                    init_dist, verbose)
        if np.all(prev_viterbi == next_viterbi):
            return transition, emission, init_dist
        else:
            prev_viterbi = next_viterbi

def baum_welch(observation, transition_seed, emission_seed, init_dist_seed,
               verbose=False):
    '''
    returns the maximum likelihood estimate of the parameters of a HMM.
    
    Parameters
    ----------
    observation: integer array of shape (T,)
    observation[i] is the observation on time i.
    
    transition_seed: float array of shape (S, S)
    transition_seed[i, j] is the transition probability of transiting from state i to state j.
    
    emission_seed: float array of shape (S, O)
    emission_seed[i, j] is the probability of observing j from state i.
    
    init_dist_seed: float array of shape (S,)
    init_dist_seed[i] is the initial probability of being state i.
    
    Returns
    -------
    transition: float array of shape (S, S)
    transition[i, j] is the transition probability of transiting from state i to state j.
    
    emission: float array of shape (S, O)
    emission[i, j] is the probability of observing j from state i.
    
    init_dist: float array of shape (S,)
    init_dist[i] is the initial probability of being state i.
    '''

    # input parameter validation
    if observation.dtype != int:
        raise ValueError('Condition observation_seed.dtype == int not met.')

    n_states, n_obs = emission_seed.shape
    if transition_seed.shape != (n_states,) * 2:
        raise ValueError('')
    if init_dist_seed.shape != (n_states,):
        raise ValueError('')

    if not observation.max() < n_obs:
        raise ValueError('')
    if not hp.array_equiv(hp.asmparray(transition_seed,
                                       ap.iv.mpf).sum(axis=-1), 1):
        raise ValueError('')
    if not hp.array_equiv(hp.asmparray(emission_seed,
                                       ap.iv.mpf).sum(axis=-1), 1):
        raise ValueError('')
    if not hp.array_equiv(hp.asmparray(init_dist_seed, ap.iv.mpf).sum(), 1):
        raise ValueError('')

    # forward procedure
    n_times = len(observation)
    for_probs = hp.zeros((n_times, n_states))
    for_probs[0] = emission_seed[:, observation[0]] * init_dist_seed
    if verbose:
        print('Proceding forward')
    with tqdm.trange(1, n_times, unit='frame', disable=not verbose) as pbar:
        for time in pbar:
            for state in range(n_states):
                for_probs[time, state] = (np.dot(for_probs[time - 1],
                                                 transition_seed[:, state]) *
                                          emission_seed[state, observation[time]])
    
    # backward procedure
    back_probs = hp.zeros((n_times, n_states))
    back_probs[-1] = 1.0
    if verbose:
        print('Proceding backward')
    with tqdm.tqdm(range(n_times - 1)[::-1], unit='frame',
                   disable=not verbose) as pbar:
        for time in pbar:
            for state in range(n_states):
                back_probs[time, state] = np.dot(transition_seed[state],
                                                 back_probs[time + 1] * 
                                                 emission_seed[:, observation[time]])

    # probability
    mono_probs = for_probs * back_probs
    mono_probs /= mono_probs.sum(axis=-1, keepdims=True)

    intermediate = np.array([emission_seed[state][observation[1:]]
                             for state in range(n_states)]).T
    # need pbar
    binary_probs = (for_probs[:-1][..., np.newaxis] *
                    transition_seed[np.newaxis] *
                    back_probs[1:][:, np.newaxis] *
                    intermediate[:, np.newaxis])
    binary_probs /= binary_probs.sum(axis=-1).sum(axis=-1)[:, np.newaxis,
                                                           np.newaxis]

    # update
    init_dist = mono_probs[0]
    transition = (binary_probs.sum(axis=0) /
                  mono_probs[:-1].sum(axis=0)[:, np.newaxis])
    transition /= transition.sum(axis=-1, keepdims=True) # normalize
    indicator = np.array([observation == obs_id for obs_id in range(n_obs)])
    emission = (np.dot(indicator, mono_probs) / mono_probs.sum(axis=0)).T
                
    return transition, emission, init_dist
