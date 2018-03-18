import numpy as np
import itertools
import scipy.ndimage.filters
import skimage.feature
import multiprocessing as mp

def cluster_mask(classifiers):
    pairs_list = [tuple(zip(edges[:-1], edges[1:])) 
                  for values, edges in classifiers]
    values_list = [values for values, edges in classifiers]
    
    shape = tuple(len(pairs) for pairs in pairs_list) + (len(values_list[0]),)
    masks = np.empty(shape, bool)
    for index, pair_sequence in enumerate(itertools.product(*pairs_list)):
        mask = np.ones(len(values_list[0]), bool)
        for values, pair in zip(values_list, pair_sequence):
            left, right = pair
            mask &= (left <= values) & (values < right)
            
        tuple_index = np.unravel_index(index, shape[:-1])
        masks[tuple_index] = mask
    return masks
    
def _gaussian_filter1d(args):
    ret = np.array([scipy.ndimage.filters.gaussian_filter1d(elem, args[1])
                    for elem in args[0]])
    return ret
    
def merge(signal, reference, transform):
    zs_from_ref = [transform((elem, 0, 0))[0] for elem in range(len(reference))]
    
def temporal_continuity(arr, order):
    cell_size, time_size = arr.shape
    result = np.empty((cell_size * (2 * order + 1), time_size - 2 * order))
    for group in range(2 * order + 1):
        cell_begin, cell_end = group * cell_size, (group + 1) * cell_size
        time_begin, time_end = group, time_size - 2 * order + group
        result[cell_begin:cell_end] = arr[:, time_begin:time_end]
        
    return result

def temporal_continuity2(arr, sigma, truncate=4.0):
    cell_size, time_size = arr.shape
    lw = int(truncate * sigma + 0.5)
    result = np.empty((cell_size * (2 * lw + 1), time_size - 2 * lw))

    kernel = np.zeros(2 * lw + 1)
    kernel[lw] = 1
    kernel = scipy.ndimage.filters.gaussian_filter1d(kernel, sigma)
    for moment, group in zip(kernel, range(2 * lw + 1)):
        cell_begin, cell_end = group * cell_size, (group + 1) * cell_size
        time_begin, time_end = group, time_size - 2 * lw + group
        result[cell_begin:cell_end] = arr[:, time_begin:time_end] * moment
        
    return result

def noise_level(vec):
    denoised = scipy.ndimage.filters.gaussian_filter1d(vec, 1)
    noise = vec - denoised
    
    return np.sqrt(np.median(noise * noise))
    
def marv(arr):
    diff = arr[:, 1:] - arr[:, :-1]
    result = np.empty(diff.shape, dtype = diff.dtype)
    for index, vec in enumerate(diff):
        centered = vec - np.median(vec)
        scaled = centered / noise_level(vec)
        result[index] = scaled
        
    return result

def increased_in_row(vec):
    increased = vec[1:] > vec[:-1]
    return count_in_row(increased)

def decreased_in_row(vec):
    decreased = vec[1:] < vec[:-1]
    return count_in_row(decreased)
     
def count_in_row(vec):
    count = vec.astype(int)
    for pos in range(len(count) - 1)[::-1]:
        count[pos] += count[pos] * count[pos + 1]
    return count
            
def flow(dst_id, cells, order):
    dst_cell = cells[dst_id]
    directions = []
    for num in pllist(range(dst_id)) + list(range(dst_id + 1, len(cells))):
        src_cell = cells[num]
        dst_act = scipy.ndimage.filters.gaussian_filter1d(dst_cell.activity,
                                                          order)
        src_act = scipy.ndimage.filters.gaussian_filter1d(src_cell.activity,
                                                          order)
        
        dst_mean = dst_act[:-1]
        src_diff = src_act[1:] - src_act[:-1]
        overlap = dst_mean * src_diff / src_diff.std()
        
        direction = dst_cell.location - src_cell.location
        directions.append(direction[np.newaxis, :] * overlap[:, np.newaxis])
    return np.array(directions)
    
def compare(dst, src):
    dst_diff = dst[:, 1:] - dst[:, :-1]
    src_diff = src[:, 1:] - src[:, :-1]
    
    dst_sq = np.mean(dst_diff * dst_diff)
    src_sq = np.mean(src_diff * src_diff)
    corr = np.mean(dst_diff * src_diff) / np.sqrt(dst_sq * src_sq)
    return corr
    
def similarity_with_template(data, tpl):
    data_times = data.shape[-1]
    tpl_times = tpl.shape[-1]
    
    score = np.empty(data_times - tpl_times + 1)
    for time in range(len(score)):
        src = data[:, time:time + tpl_times]
        score[time] = compare(tpl, src)
        
    return score
    
def similarity_from_dist(data, tpl, n_shuffles):
    if len(data) != len(tpl):
        raise ValueError('len(data) == len(tpl) not met')
        
    this = similarity_with_template(data, tpl)
    dist = np.empty((n_shuffles, len(this)))
    shuffled = tpl.copy()
    for index in range(n_shuffles):
        np.random.shuffle(shuffled)
        dist[index] = similarity_with_template(data, shuffled)

    return this / dist.std(axis = 0)

def similarity_with_width(data, width):
    with mp.Pool(processes = mp.cpu_count() - 1) as pool:
        args = ((data, data[:, time:time + width])
                for time in range(data.shape[-1] - width + 1))
        result = pool.map(_similarity_zipped, args)
    return np.array(result)
    
def _similarity_zipped(args):
    return similarity_with_template(*args)
            
def portion(data, begins, width):
    result = np.empty((len(begins), len(data), width))
    for index, begin in enumerate(begins):
        result[index] = data[:, begin:begin + width]
    return result
     
def superpose(data, moment, width):
    result = np.zeros((len(data), width))
    for time in range(len(moment)):
        result += data[:, time:time + width] * moment[time]
    return result / len(moment)
    
def superstd(data, moment, template):
    result = np.zeros(template.shape)
    for time in range(len(moment)):
        diff = (data[:, time:time + template.shape[-1]] - template)
        result += diff * diff * moment[time]
    return result / len(moment)
    
def convolve(tpl, moment):
    result = np.empty((len(tpl), len(moment) + tpl.shape[-1] - 1))
    for index in range(len(tpl)):
        result[index] = np.convolve(tpl[index], moment)
    return result
      
def create_template(data, times, order, func = np.median):
    stack_buffer = np.empty((len(times), len(data), order))
    for index, time in enumerate(times):
        stack_buffer[index] = data[:, time:time + order]
        
    stacked = func(stack_buffer, axis = 0)
    return stacked
    
def similar_times(data, tmp, thres):
    num_times = data.shape[-1]
    width = tmp.shape[-1]
    
    score = np.empty(num_times - width)
    for time in range(len(score)):
        src = data[:, time:time + width]
        score[time] = compare(tmp, src)

#    candidates = []
    self_at = score.argmax()
    others = np.r_[score[:self_at], score[self_at + 1:]]
    threshold = others.std() * thres
    candidates = skimage.feature.peak_local_max(score, width // 2, threshold)
#    for similar, time_value in itertools.groupby(enumerate(score),
#                                                 lambda i: i[-1] > threshold):
#        if similar:
#            times, values = zip(*time_value)
#            candidate = times[np.argmax(values)]
#            candidates.append(candidate)
    
    return candidates.reshape((candidates.size,)), score
    
def analyze(data, tmp_begin, tmp_end, thres = 3):
    tmp = data[:, tmp_begin:tmp_end]
    width = tmp_end - tmp_begin
    
    offsets = []
    while True:
        candidates = similar_times(data, tmp, thres)
        if len(offsets) != 0:
            futures = list(itertools.dropwhile(lambda i: i <= offsets[-1],
                                               candidates))
            if len(futures) != 0:
                offsets.append(futures[0])
            else:
                break
        else:
            if len(candidates) != 0:
                offsets.append(candidates[0])
            else:
                break
        
        new_tmp_begin = offsets[-1]
        tmp = data[:, new_tmp_begin:new_tmp_begin + width]
        
    return offsets
