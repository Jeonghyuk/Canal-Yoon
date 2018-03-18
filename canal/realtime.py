import numpy as np

def convert(time_arr, height_arr, height_size):
	"""
	Returns converted time array of an input based on heights.
	"""
	if len(time_arr) != len(height_arr):
		raise Exception('Height array and time array include different number of elements.')

	if isinstance(time_arr, list):
		ret_lst = []
		for height, pack in zip(height_arr, time_arr):
			ret_lst.append(list(np.array(pack) * (height_size + 2) + height))
		return ret_lst
	else:
		return time_arr * (height_size + 2) + height_arr[:, np.newaxis]

def interpolate(src, height_arr, height_size):
	"""
	Returns interpolated time series.
	"""
	cell_size, time_size = src.shape
	dst_pos = np.arange(time_size * (height_size + 2))
	interped = np.empty([cell_size, len(dst_pos)])
	for cell in range(cell_size):
		src_pos = np.arange(time_size) * (height_size + 2) + height[cell]
		interped[cell] = np.interp(dst_pos, src_pos, src[cell])

	return interped
