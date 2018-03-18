import numpy as np
import canal.local
import canal.esti
import scipy.ndimage.filters

def normalize(vec):
	# calc window width and noise
	prev_min_noise = 0
	diff = vec[1:] - vec[:-1]
	for width in range(1, diff.size):
		noises = canal.local.std_v2(diff, width)	# std time-series with size width
		min_noise = min(noises)	# min of the time-series
		if min_noise < prev_min_noise:
			window_width = width

			hist, edges = np.histogram(noises, 1024)
			max_index = hist.argmax()
			noise = edges[max_index:max_index + 2].mean()

			del noises, prev_min_noise, min_noise, hist, edges, max_index
			break
		else:
			prev_min_noise = min_noise

	# calc slopes
	slopes = np.empty(vec.size)
	#slopes[window_width 

	# min of vec
	min_vec = -canal.local.max_v2(-vec, window_width)
	base = min_vec.copy()
	activity = np.zeros(vec.size, dtype = bool)
	threshold = 3 * noise
	for time in range(1, vec.size):
		if vec[time] > base[time - 1] + threshold:	# active
			activity[time] = True
			base[time] = base[time - 1]
		else:	# inactive
			activity[time] = False
			if activity[time - 1]:	# turned off
				base[time] = 1
			else:	# no transition
				base[time] = min(min_vec[time] + threshold, vec[time])

	base = np.empty(vec.size)
	base[:window_width] = vec[:window_width].mean()
	for time in range(window_width, vec.size):
		vals = vec[time - window_width:time]
		mask = ~activity[time - window_width:time]
		if mask.any():
			base[time] = vals[mask].mean()
		else:
			base[time] = base[time - 1]

	return base

def test(vec):
	time_size = vec.size

	# calc noise level
	diff = vec[1:] - vec[:-1]
	noise = diff.std()

	# calc seeds
	# using local maxima within a window size 3
	seeds = canal.local.argzero(diff)[1]

	# process
	while True:
		# calc the heights of the barriers
		scores = np.empty(len(seeds) - 1)
		for index in range(len(scores)):
			left_seed, right_seed = seeds[index], seeds[index + 1]
			middle = vec[left_seed + 1:right_seed]
			scores[index] = min(vec[left_seed], vec[right_seed]) - middle.min()

		# merge the seeds
		mask = np.ones(len(seeds), dtype = bool)
		for score in scores:
			if score < noise:
				pass
	
	
def find(vec, dbg = False):
	time_size = vec.size
	mag, fold = 4, 2

	# 1st estimate of filter: global filter width
	filter_width = canal.esti.filter_width(vec, time_size // mag)

	# 2nd estimate of filter: filter widths of the portions
	window_width, window_step = filter_width * mag, filter_width * mag // fold
	filter_widths = []
	for window in range(1, time_size // window_step - 1):
		left = window_step * window
		portion = vec[left:left + window_width]
		width = canal.esti.filter_width(portion, window_width // 2)
		filter_widths.append(width)
	filter_width = np.median(filter_widths)

	def extrema_from_noise(vec, smoothed, ratio):
		# evaluate the noise level
		global_noise = np.std(vec - smoothed)
		thres = global_noise * ratio

		# find local minima and maxima
		diff = smoothed[1:] - smoothed[:-1]
		minima, maxima = canal.local.argzero(diff)
		minima, maxima = list(minima), list(maxima)
		if minima[0] > maxima[0]:
			minima.insert(0, 0)
		if minima[-1] < maxima[-1]:
			minima.append(time_size - 1)

		# merge local maxima
		while len(maxima) > 1:
			for index in range(len(maxima) - 1):
				lpeak, rpeak = maxima[index], maxima[index + 1]
				mid = minima[index + 1]
				lval, mval, rval = smoothed[lpeak], smoothed[mid], smoothed[rpeak]
				if min(lval - mval, rval - mval) < thres:
					if lval < rval: # eliminate the left peak
						maxima.pop(index)
						mid_cand = index
					else:
						maxima.pop(index + 1)
						mid_cand = index + 2

					try:
						if mval < smoothed[minima[mid_cand]]: # eliminate the candidate
							minima.pop(mid_cand)
						else:
							minima.pop(index + 1)
					except IndexError:
						minima.pop(index + 1)
					break
			else:
				break

		return minima, maxima

	# use raw time series
	smoothed = scipy.ndimage.filters.gaussian_filter1d(vec, filter_width)
	minima, maxima = extrema_from_noise(vec, smoothed, 2)

	# estimate of trend: interpolate the onset values
	base = np.interp(range(time_size), minima, smoothed[minima])
	detrended = vec / base
	smoothed = scipy.ndimage.filters.gaussian_filter1d(detrended, filter_width)
	minima, maxima = extrema_from_noise(detrended, smoothed, 1)

	# intensity based detection
	def extrema_from_intensity(vec, peaks):
		thres = vec[peaks].min()

		# find local minima and maxima
		diff = vec[1:] - vec[:-1]
		minima, maxima = canal.local.argzero(diff)
		minima, maxima = list(minima), list(maxima)
		if minima[0] > maxima[0]:
			minima.insert(0, 0)
		if minima[-1] < maxima[-1]:
			minima.append(time_size - 1)

		# find peaks
		new_minima, new_maxima = set(), set()
		for index, peak in enumerate(maxima):
			left, right = minima[index], minima[index + 1]
			if peak not in peaks:
				new_val = (peak - left) * (vec[right] - vec[left]) / (right - left) + vec[left]
				if new_val < thres:
					continue

			new_maxima.add(peak)
			new_minima.update((left, right))

		return sorted(new_minima), sorted(new_maxima)

	minima, maxima = extrema_from_intensity(smoothed, maxima)
	base = np.interp(range(time_size), minima, smoothed[minima])
	detrended = detrended / base - 1
	normalized = smoothed / base - 1

	# correct the peaks: local maxima around the peaks
	def argzero(vec):
		for index, val in enumerate(vec):
			if val < 0:
				break
		return index

	peaks = []
	for peak in maxima:
		left = peak - argzero(detrended[peak::-1])
		right = peak + argzero(detrended[peak:])
		portion = detrended[left:right]
		if len(portion) > 0:
			corrected = left + portion.argmax()
			peaks.append(corrected)

	if dbg == True:
		import matplotlib.pyplot as plt
		plt.plot(base)
		plt.plot(smoothed)
		plt.figure()
		plt.plot(detrended)
		for index, peak in enumerate(peaks):
			plt.plot([peak] * 2, [0, detrended[peak]], color = 'black')
			#plt.plot(bases[index], sharp[np.array(bases[index])], color = 'red')
	
	return peaks, normalized

def synchronized(dst, src, sync_width, sync_limit = None):
	if sync_limit == None:
		hist_min, hist_max = dst[0] - src[-1], dst[-1] - src[0]
	else:
		hist_min, hist_max = -sync_limit, sync_limit

	dst_size, src_size = len(dst), len(src)
	diffs = []
	hist = np.zeros(hist_max - hist_min + 1, dtype = int)
	for dst_index, dst_elem in enumerate(dst):
		for src_index, src_elem in enumerate(src):
			diff = dst_elem - src_elem
			if hist_min <= diff and diff <= hist_max:
				hist[diff - hist_min] += 1
				diffs.append((diff, dst_index, src_index))

	count_sync = canal.local.sum(hist, sync_width)
	offset = count_sync.argmax() + hist_min
	diffs = sorted(diffs, key = lambda i: abs(i[0] - offset))

	sync_pairs = []
	sync_range = (offset - sync_width * 3, offset + sync_width * 3 + 1)
	for diff, dst_index, src_index in diffs:
		if sync_range[0] <= diff and diff <= sync_range[1]:
			for search, dummy in sync_pairs:
				if search == dst_index:
					break
			else:
				sync_pairs.append((dst_index, src_index))

	return sync_pairs
