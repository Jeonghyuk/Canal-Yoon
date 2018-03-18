import numpy as np
import math
import itertools
import scipy.ndimage.filters
import canal.registration
import canal.local
	
def interp(references, ratio = 5):
	stacked = references.mean(axis = 0)
	height_size, row_size, col_size = stacked.shape
	interped_size = (height_size - 1) * ratio + 1

	# calc frequency domain
	ffted = np.empty(stacked.shape, dtype = complex)
	for row in range(row_size):
		for col in range(col_size):
			ffted[:, row, col] = np.fft.fft(stacked[:, row, col])

	# zero-inserted
	zeroed = np.zeros((interped_size, row_size, col_size), dtype = complex)
	zeroed[:height_size // 2 + 1] = ffted[:height_size // 2 + 1]
	zeroed[-(height_size // 2):] = ffted[-(height_size // 2):]

	# interpolate
	interped = np.empty(zeroed.shape)
	for row in range(row_size):
		for col in range(col_size):
			interped[:, row, col] = np.fft.ifft(zeroed[:, row, col]).real

	return interped

def splice(references, signals):
	ref_offsets = offsets_static(references)
	sig_offsets = offsets_dynamic(signals)
	sig_offsets += ref_offsets[-1]

	aligned_ref = align(references, np.concatenate((ref_offsets, sig_offsets)))
	aligned_sig = align(signals, np.concatenate((sig_offsets, ref_offsets)))

	return aligned_ref, aligned_sig

def offsets_static(arr):
	time_size, height_size, row_size, col_size = arr.shape
	offsets = np.empty((time_size, 2), dtype = int)
	for time in range(0, time_size):
		print('Calculating offsets: {} out of {}'.format(time, time_size), end = '\r')
		delta = np.zeros((row_size, col_size))
		for height in range(0, height_size):
			src = arr[0, height]
			dst = arr[time, height]
			delta += canal.registration.translation(dst, src, crit = 0, distribution = True)

		row, col = np.unravel_index(delta.argmax(), delta.shape)
		row = row if row < row_size // 2 else row - row_size
		col = col if col < col_size // 2 else col - col_size
		offsets[time] = (row, col)

	return offsets

def align(arr, offsets):
	time_size, height_size, row_size, col_size = arr.shape
	top_left = offsets.min(axis = 0)
	bottom_right = np.array((row_size, col_size)) - offsets.max(axis = 0)
	new_size = bottom_right - top_left
	aligned = np.empty((time_size, height_size, new_size[0], new_size[1]), dtype = arr.dtype)
	for time in range(time_size):
		start, end = top_left + offsets[time], bottom_right + offsets[time]
		aligned[time] = arr[time, :, start[0]:end[0], start[1]:end[1]]

	return aligned

def offsets_dynamic(arr, width = 10, crit = 0.02):
	time_size, height_size, row_size, col_size = arr.shape

	# find time frame of inactive state
	activity = np.array([elem.sum() for elem in arr])
	inactive_times = canal.local.argmax_v2(-activity, width)

	# stack
	inactive_size = len(inactive_times)
	stacked = np.empty((inactive_size, height_size, row_size, col_size))
	for index, time in enumerate(inactive_times):
		stacked[index] = arr[time - width // 2:time + width // 2].mean(axis = 0)
		stacked[index] /= stacked[index].mean()

	# calc offsets
	inactive_offsets = np.zeros((inactive_size, 2), dtype = int)
	src_index, index = 0, 0
	while index < inactive_size:
		print('Calculating offsets: {} out of {}'.format(index, inactive_size), end = '\r')
		delta = np.zeros((row_size, col_size))
		for height in range(0, height_size):
			src = stacked[src_index, height]
			dst = stacked[index, height]
			delta += canal.registration.translation(dst, src, crit = 0, distribution = True)

		row, col = np.unravel_index(delta.argmax(), delta.shape)
		score = delta[row, col] / height_size
		if score < crit:
			if src_index == index - 1:
				src_index = index
				index += 1
			else:
				src_index = index - 1
				inactive_offsets[src_index + 1:] = inactive_offsets[src_index]
		else:
			row = row if row < row_size // 2 else row - row_size
			col = col if col < col_size // 2 else col - col_size
			inactive_offsets[index] += (row, col)
			index += 1

	# interpolate offsets
	offsets = np.empty((time_size, 2))
	for col in range(2):
		offsets[:, col] = np.interp(np.arange(time_size), inactive_times, inactive_offsets[:, col]).round().astype(int)

	return offsets

def masks(arr, radius = (1.5, 5.5, 5.5)):
	'''
	# temporary shrink
	shrink = 5
	new_arr = np.empty((arr.shape[0] // shrink,) + arr.shape[1:])
	for index in range(len(new_arr)):
		new_arr[index] = arr[shrink * index:shrink * (index + 1)].mean(axis = 0)
	arr = new_arr
	'''
	time_size, height_size, row_size, col_size = arr.shape

	# True filled kernel of elliptical shape
	def elliptical_kernel(radius):
		denom = np.array(radius)
		center = np.floor(denom + 0.5).astype(int)
		diameter = center * 2 + 1
		kernel = np.zeros(diameter, dtype = bool)

		for index in itertools.product(*[range(elem) for elem in diameter]):
			diff = np.array(index) - center
			normalized = diff / denom
			if np.dot(normalized, normalized) < 1:
				kernel[index] = True
		return kernel

	# get the indices at which the two input array have same value
	def argequal(dst, src):
		#index_array = np.array(list(itertools.product(*[range(elem) for elem in src.shape]))).reshape(src.shape + (len(src.shape),))
		#return index_array[src == dst]
		index_array = np.arange(src.size).reshape(src.shape)
		return [np.unravel_index(flat_index, src.shape) for flat_index in index_array[src == dst]]

	# use gaussian for each time frame
	window = elliptical_kernel(radius)
	maxima = [[] for time in range(time_size)]
	filtered = np.empty(arr.shape)
	filter_radius = tuple(elem / 2 for elem in radius)
	for time in range(time_size):
		print('Scanning time frames: frame {} out of {}'.format(time, time_size), end = '\r')
		# get local minima of gaussian filtered movie
		filtered[time] = scipy.ndimage.filters.gaussian_filter(arr[time], filter_radius)
		max_filtered = scipy.ndimage.filters.maximum_filter(filtered[time], footprint = window)
		maxima[time] = argequal(max_filtered, filtered[time])

	# get points which have neighbors (return points of dst)
	def coupled(dst, src, rad):
		points = []
		print(len(dst), len(src))
		for test, src_elem in enumerate(src):
			diffs = np.array(dst) - np.array(src_elem)
			normed = diffs / rad
			dist_sq = np.sum(normed * normed, axis = 1)
			cand_arg = dist_sq.argmin()
			distance = math.sqrt(dist_sq[cand_arg])
			if distance < 1:
				points.append(dst[cand_arg])
		return points

	# compare the minima of different time frames with each other
	def compress(lst, rad, origin = 0):
		acc = lst[origin:].copy()
		while len(acc) > 1:
			new = []
			for index in range(len(acc) - 1):
				dst, src = acc[index], acc[index + 1]
				new.append(coupled(dst, src, rad))
			acc = new

		acc = lst[:origin] + acc
		while len(acc) > 1:
			new = []
			for index in range(len(acc) - 1):
				dst, src = acc[index + 1], acc[index]
				new.append(coupled(dst, src, rad))
			acc = new
		return acc[0]

	centers = compress(maxima, radius, time_size // 2)
	window_center = tuple((elem - 1) // 2 for elem in window.shape)
	stacked = arr.mean(axis = 0)
	masks = []

	def trim(lower, upper, src = stacked):
		heights, rows, cols = [(l, u) for l, u in zip(lower, upper)]
		return src[heights[0]:heights[1], rows[0]:rows[1], cols[0]:cols[1]]

	for center in centers:
		lower = tuple(max(c - w, 0) for c, w in zip(center, window_center))
		upper = tuple(min(c + w + 1, l) for c, w, l in zip(center, window_center, stacked.shape))
		mask = trim(lower, upper).copy()
		mask /= mask.sum()
		masks.append((lower, upper, center, mask))

	return masks

def signals(arr, masks, reference_nheight):
	time_size, height_size, row_size, col_size = arr.shape

	# in reference movie
	height_gap = 1 + (reference_nheight - height_size) // (height_size - 1)
	valid_heights = range(0, reference_nheight, height_gap)

	positions, fluos = [], []
	for index, (lower, upper, center, mask) in enumerate(masks):
		print('Extracting: mask {} out of {}'.format(index, len(masks)), end = '\r')
		mask_heights = range(lower[0], upper[0])
		try:
			heights_in_mask, heights_in_signal = zip(*[(mheight, cheight // height_gap) for mheight, cheight in enumerate(mask_heights) if cheight in valid_heights])
		except ValueError:	# empty list
			pass
		else:
			included_signals = arr[:, heights_in_signal, lower[1]:upper[1], lower[2]:upper[2]]
			included_masks = mask[heights_in_mask]
			masked = included_signals * included_masks[np.newaxis, :]
			fluo = np.mean(np.mean(np.mean(masked, axis = -1), axis = -1), axis = -1)
			position = (int(center[0] / height_gap),) + center[1:]
			positions.append(position)
			fluos.append(fluo)

	return np.array(fluos), np.array(positions)
