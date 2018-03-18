import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import pickle
import sklearn.decomposition
import math
import random
import sklearn.cluster
import matplotlib.cm as cm

import canal.esti
import canal.core

def cluster(normalized, n_sclusters, n_tclusters, time_range = None):
	cell_size, time_size = normalized.shape
	if time_range == None:
		time_range = (0, time_size)

	# spatial clustering
	kmeans = sklearn.cluster.KMeans(init = 'k-means++', n_clusters = n_sclusters)
	kmeans.fit(normalized)
	#kmeans = sklearn.cluster.AffinityPropagation()
	#kmeans.fit(normalized)
	spatial_labels = kmeans.labels_.copy()
	#n_sclusters = len(set(spatial_labels))

	# temporal clustering
	kmeans = sklearn.cluster.KMeans(init = 'k-means++', n_clusters = n_tclusters)
	kmeans.fit(normalized.T)
	#kmeans = sklearn.cluster.AffinityPropagation()
	#kmeans.fit(normalized.T)
	temporal_labels = kmeans.labels_.copy()
	#n_tclusters = len(set(temporal_labels))
	
	spatial_groups = []
	for group_number in range(n_sclusters):
		cells = [cell for cell, g in enumerate(spatial_labels) if g == group_number]
		spatial_groups.append(normalized[cells])

	#leaders = np.array([elem.mean(axis = 0) for elem in spatial_groups])
	leaders = np.array([elem[0] for elem in spatial_groups])
	#leaders /= leaders.std(axis = 1)[:, np.newaxis]

	color_template = canal.out.get_colors(n_tclusters)
	tcolors = [color_template[g] for g in temporal_labels[time_range[0]:time_range[1]]]

	pca = sklearn.decomposition.PCA(n_components = 3)
	pca.fit(normalized.T)
	shrink = pca.transform(normalized.T)[time_range[0]:time_range[1]].T
	fig = plt.figure()
	ax = fig.add_subplot(111, projection = '3d')
	ax.scatter(shrink[0], shrink[1], shrink[2], color = tcolors)
	ax.plot(shrink[0], shrink[1], shrink[2], color = 'black')
	plt.title('All neurons (N = {})'.format(cell_size))
	plt.show()
	
	pca = sklearn.decomposition.PCA(n_components = 3)
	pca.fit(leaders.T)
	shrink = pca.transform(leaders.T)[time_range[0]:time_range[1]].T
	fig = plt.figure()
	ax = fig.add_subplot(111, projection = '3d')
	for index in range(time_range[1] - time_range[0] - 1):
		ax.plot(shrink[0, index:index + 2], shrink[1, index:index + 2], zs = shrink[2, index:index + 2], color = tcolors[index])
	plt.title('Leader neurons (N = {})'.format(n_sclusters))
	plt.show()

	def add_frames(ax, spatial_labels, temporal_labels = temporal_labels[time_range[0]:time_range[1]]):
		spatial_rect = [0, 0, 0.01, 1]
		temporal_rect = [0, 0, 1, 0.01]

		if spatial_labels != None:
			scolors = (spatial_labels / (spatial_labels.max() + 1)).reshape(len(spatial_labels), 1)
			saxes = add_subplot_axes(ax, spatial_rect)
			saxes.imshow(scolors, interpolation = 'none', cmap = cm.hsv, aspect = 'auto')
			plt.xticks([])
			plt.yticks([])

		if temporal_labels != None:
			tcolors = (temporal_labels / (temporal_labels.max() + 1)).reshape(1, len(temporal_labels))
			taxes = add_subplot_axes(ax, temporal_rect)
			taxes.imshow(tcolors, interpolation = 'none', cmap = cm.hsv, aspect = 'auto')
			plt.xticks([])
			plt.yticks([])

	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.imshow(normalized[np.argsort(spatial_labels), time_range[0]:time_range[1]], aspect = 'auto', interpolation = 'none')
	plt.xlabel('Time [Frame]')
	plt.ylabel('Cell Index')
	plt.title('All neurons (N = {})'.format(cell_size))
	add_frames(ax, spatial_labels = np.array(sorted(spatial_labels)))
	

	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.imshow(leaders[:, time_range[0]:time_range[1]], aspect = 'auto', interpolation = 'none')
	plt.xlabel('Time [Frame]')
	plt.ylabel('Cell Index')
	plt.title('Leader neurons (N = {})'.format(n_sclusters))
	add_frames(ax, spatial_labels = None)
	

def add_subplot_axes(ax,rect,axisbg='w'):
	fig = plt.gcf()
	box = ax.get_position()
	width = box.width
	height = box.height
	inax_position  = ax.transAxes.transform(rect[0:2])
	transFigure = fig.transFigure.inverted()
	infig_position = transFigure.transform(inax_position)    
	x = infig_position[0]
	y = infig_position[1]
	width *= rect[2]
	height *= rect[3]
	subax = fig.add_axes([x,y,width,height],axisbg=axisbg)
	return subax

def test(normalized):
	colors = canal.out.get_colors(normalized.shape[0])
	fig = plt.figure()
	ax = fig.add_subplot(111)
	
	plt.imshow(normalized, aspect = 'auto', interpolation = 'none')
	plt.xlabel('Time [Frame]')
	plt.ylabel('Cell Index')
	plt.title('All neurons (N = {})'.format(cell_size))

def count_peaks(peaks, time_size):
	cell_size = len(peaks)
	count = np.zeros(time_size, dtype = int)

	for cell in range(cell_size):
		for peak in peaks[cell]:
			count[peak] += 1

	return count

def states(peaks, time_size, positions, dimension, smooth = 10):
	# remove cells have few peaks
	valid_cells = [cell for cell, elem in enumerate(peaks) if len(elem) > 50]
	peaks = [elem for cell, elem in enumerate(peaks) if cell in valid_cells]
	positions = positions[valid_cells]

	left_border, right_border = dimension[2] / 5, dimension[2] * 4 / 5
	left_cells = [cell for cell, pos in enumerate(positions) if pos[2] < left_border]
	right_cells = [cell for cell, pos in enumerate(positions) if pos[2] > right_border]

	front_border, rear_border = dimension[1] / 5, dimension[1] * 4 / 5
	front_cells = [cell for cell, pos in enumerate(positions) if pos[1] < front_border]
	rear_cells = [cell for cell, pos in enumerate(positions) if pos[1] > rear_border]

	cell_groups = [left_cells, right_cells, front_cells, rear_cells]
	group_counts = []
	for group in cell_groups:
		group_peaks = [elem for cell, elem in enumerate(peaks) if cell in group]
		group_counts.append(np.correlate(count_peaks(group_peaks, time_size), np.ones(smooth)))

	profiles = np.c_[tuple(group_counts)]
	result = profiles[:, 1:].copy()
	result[:, 1] -= profiles[:, 0]
	result = result.T
	#plt.imshow(profiles, aspect = 'auto')

	# test raw PCA
	pca_solver = sklearn.decomposition.PCA(n_components = 3).fit(profiles)
	#result = pca_solver.transform(profiles).T
	print(result.shape)

	fig = plt.figure()
	ax = fig.add_subplot(111, projection = '3d')
	ax.plot(result[0], result[1], result[2])
	plt.show()

	return profiles

def traj(normalized, positions, dimension, time_scale = 10):
	lr_asym = np.cos(positions[:, 1] * (np.pi / dimension[1]))
	bf_asym_spat = np.cos(positions[:, 2] * (np.pi / dimension[2]))
	bf_asym_temp = np.cos(np.linspace(0, np.pi / 2, time_scale))

	traj_sym = np.correlate(normalized.sum(axis = 0), np.ones(time_scale))
	traj_lr = np.correlate(np.sum(normalized * lr_asym[:, np.newaxis], axis = 0), np.ones(time_scale))
	traj_bf = np.correlate(np.sum(normalized * bf_asym_spat[:, np.newaxis], axis = 0), bf_asym_temp)

	trajectory = np.c_[traj_bf, traj_lr, traj_sym]

	fig = plt.figure()
	ax = fig.add_subplot(111, projection = '3d')
	ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2])
	plt.xlabel('forward/backward')
	plt.ylabel('turning')
	plt.zlabel('activity')
	plt.show()

	return trajectory

def find_waves(src_peaks, normalized, progress = True):
	cell_size, time_size = normalized.shape
	# find global activity windows roughly
	activity = np.mean(normalized / normalized.std(axis = 1)[:, np.newaxis], axis = 0)
	activity = canal.esti.smoothen(activity, len(activity) // 8)

	def local_minima(vec):
		diff = vec[1:] - vec[:-1]
		return canal.local.argzero(diff)[0]

	wave_borders = local_minima(activity)

	plt.figure()
	plt.plot(activity)

	# shift borders to owner's local minima
	def slide(point, geometry):
		slopes = geometry[1:] - geometry[:-1]
		local_min = canal.local.argzero(slopes)[0]
		if slopes[min(len(slopes) - 1, point)] < 0:
			side = local_min[local_min > point]
			if len(side) == 0:
				return len(geometry) - 1
			else:
				return side[0]
		else:
			side = local_min[local_min <= point]
			if len(side) == 0:
				return 0
			else:
				return side[-1]

	# find peaks using individual activity windows
	borders = [[slide(border, normalized[cell]) for cell in range(cell_size)] for border in wave_borders]

	def wave_peaks(left_borders, right_borders, arr = normalized, src_peaks = src_peaks):
		cell_size = len(src_peaks)
		peaks = np.empty(cell_size, dtype = int)
		for cell in range(cell_size):
			left, right = left_borders[cell], right_borders[cell]
			in_range = [peak for peak in src_peaks[cell] if left <= peak < right]
			try:
				peaks[cell] = in_range[arr[cell, in_range].argmax()]
			except ValueError:
				peaks[cell] = -1
		return peaks

	# check whether the waves is splited
	def merge_peaks(peaks, borders, arr = normalized, rel_thres = 4):
		left_borders, right_borders = borders
		merged = wave_peaks(left_borders, right_borders)

		both_peak = np.array([elem != -1 for elem in peaks]).all(axis = 0)
		score = 0
		for cell, check in enumerate(both_peak):
			if check:
				lsig, rsig = arr[cell, peaks[0][cell]], arr[cell, peaks[1][cell]]
				ratio = lsig / rsig
				if 1 / rel_thres > ratio or ratio > rel_thres:
					score += 1
			else:
				score += 1
		score /= len(both_peak)
		return merged, score

	# create seed waves
	peaks = []
	for wave in range(len(borders) - 1):
		peaks.append(wave_peaks(borders[wave], borders[wave + 1]))

	# merge waves
	crit_score = 3 / 4
	while True:
		if progress == True:
			print('Merging waves: {} waves'.format(len(peaks)), end = '\r')
		new_peaks = []
		scores = []
		for face in range(len(peaks) - 1):
			left_peaks, right_peaks = peaks[face], peaks[face + 1]
			left_borders, right_borders = borders[face], borders[face + 2]
			merged, score = merge_peaks((left_peaks, right_peaks), (left_borders, right_borders))
			new_peaks.append(merged)
			scores.append(score)

		top_score = max(scores)
		if top_score > crit_score:
			merge_face = scores.index(top_score)
			peaks.pop(merge_face)
			peaks.pop(merge_face)
			peaks.insert(merge_face, new_peaks[merge_face])
			borders.pop(merge_face + 1)
		else:
			break
	wave_size = len(peaks)
	peaks = np.array(peaks, dtype = int).T
	#if progress == True:
	#	print(_erase_line, end = '')

	# eliminate the waves with low activity
	if progress == True:
		print('Estimating distribution of the waves: Assume gaussian', end = '\r')
	try:
		peak_counts = np.sum(peaks != -1, axis = 0)
		print(peak_counts)
		peak_count, peak_count_std = canal.stats.gaussian_profile(peak_counts)
		print('{} += {}'.format(peak_count, peak_count_std))
	except Exception:
		print('peaks per wave vary too much')
	else:
		peak_count_thres = peak_count - 4 * peak_count_std
		valid_waves = [wave for wave, elem in enumerate(peak_counts) if elem > peak_count_thres]

		peaks = peaks[:, valid_waves]
		wave_size = len(valid_waves)
	#if progress == True:
	#	print(_erase_line, end = '')

	# calc pseudo-phases

	return peaks
	clean_peaks = np.array([elem for elem in peaks if not np.any(elem == -1)], dtype = int)
	clean_size = len(clean_peaks)
	if progress == True:
		print('Evaluating pseudo phases: {} cells involved'.format(clean_size))
	pseudo_phases = np.empty([wave_size * 2, clean_size])
	for wave in range(wave_size):
		pseudo_phases[wave] = clean_peaks[:, wave].argsort() / clean_size - 0.5

	# insert reversed phases
	pseudo_phases[wave_size:] = -pseudo_phases[:wave_size]		

	# define metric on pseudo-phase space
	def gen_metric(dim):
		vec = np.arange(dim) / (dim - 1)
		diff = vec - vec[::-1]
		_max = math.sqrt(np.dot(diff, diff))
		def _metric(dst, src, norm = _max):
			diff = dst - src
			return math.sqrt(np.dot(diff, diff)) / norm
		return _metric
	metric = gen_metric(clean_size)

	# calc distances
	distances = np.empty([len(pseudo_phases)] * 2)
	for dst_wave, dst_phases in enumerate(pseudo_phases):
		for src_wave, src_phases in enumerate(pseudo_phases):
			distances[dst_wave, src_wave] = metric(dst_phases, src_phases)

	plt.figure()
	_x = np.arange(clean_size) / (clean_size - 1)
	_y = np.sqrt(1 - _x * _x)
	plt.plot(_x, _y)
	# group test
	backward_template = np.arange(clean_size) / clean_size - 0.5
	forward_template = backward_template[::-1].copy()
	border_ref = 0.41 #metric(forward_template, backward_template) / 3
	wave_profiles = np.empty(wave_size, dtype = int)
	for wave, phase in enumerate(pseudo_phases[:wave_size]):
		forward_distance = metric(phase, forward_template)
		backward_distance = metric(phase, backward_template)
		plt.plot(forward_distance, backward_distance, 'ko')
		plt.text(forward_distance, backward_distance, str(wave))
		if forward_distance < border_ref:
			wave_profiles[wave] = 0
		elif backward_distance < border_ref:
			wave_profiles[wave] = 1
		else:
			wave_profiles[wave] = -1
	#if progress == True:
	#	print(_erase_line, end = '')

	n_components = 3
	# create the kernel for kernel PCA
	def centralize(kernel, n_components = n_components):
		return kernel
		factor = np.empty(kernel.shape)
		factor.fill(1 / n_components)
		return kernel - np.dot(factor, kernel) - np.dot(kernel, factor) + np.dot(np.dot(factor, kernel), factor)

	# execute kernel PCA
	kernel = centralize(distances)
	pca_solver = sklearn.decomposition.PCA(n_components = n_components).fit(kernel)
	result = pca_solver.transform(kernel)
	res = result
	'''
	# test raw PCA
	pca_solver = sklearn.decomposition.PCA(n_components = n_components).fit(psudo_phases)
	result = pca_solver.transform(psudo_phases)
	res = result
	'''

	wave_color_template = ('yellow', 'red', 'green')
	wave_colors = np.array([wave_color_template[profile + 1] for profile in wave_profiles], dtype = str)
	# show results (temp)
	if n_components == 3:
		from mpl_toolkits.mplot3d import Axes3D
		fig = plt.figure()
		ax = Axes3D(fig)
		ax.scatter(res[:wave_size, 0], res[:wave_size, 1], res[:wave_size, 2], color = wave_colors)
		for wave, elem in enumerate(res[:wave_size]):
			ax.text(*elem, s = '#{}'.format(wave))
		#ax.scatter(res[wave_size:, 0], res[wave_size:, 1], res[wave_size:, 2], color = 'red')
		#for wave, elem in enumerate(res[wave_size:]):
		#	ax.text(*elem, s = '#{}-'.format(wave))
	elif n_components == 2:
		plt.figure()
		plt.plot(res[:wave_size, 0], res[:wave_size, 1], 'o')
		for wave, elem in enumerate(res[:wave_size]):
			plt.text(*elem, s = '#{}'.format(wave))
		plt.plot(res[wave_size:, 0], res[wave_size:, 1], 'ro')
		for wave, elem in enumerate(res[wave_size:]):
			plt.text(*elem, s = '#{}-'.format(wave))
	elif n_components == 1:
		plt.figure()
		plt.plot(res[:wave_size], np.zeros(len(res) // 2), 'o')
		for wave, elem in enumerate(res[:wave_size]):
			plt.text(elem, 0, s = '#{}'.format(wave))
		plt.plot(res[wave_size:], np.zeros(len(res) // 2), 'ro')
		for wave, elem in enumerate(res[wave_size:]):
			plt.text(elem, 0, s = '#{}-'.format(wave))

	return peaks, wave_profiles
	#if result == True:
	#	if len(given_cells) != len(valid_cells):
	#		print('Inactive cells: {}'.format(', '.join(str(elem) for elem in set(given_cells).difference_update(valid_cells))))


def find_trajectory(peaks, time_size, interval, term, n_components):
	cell_size = len(peaks)

	# peaks before the time frame
	init = np.arange(-cell_size - 1, -1)
	random.shuffle(init)
	past = np.empty((time_size, cell_size), dtype = int)
	future = np.empty((time_size, cell_size), dtype = int)
	for cell in range(cell_size):
		past[:, cell] = init[cell]
		future[:, cell] = init[cell]
		for peak in peaks[cell]:
			past[peak:, cell] = peak
		for peak in peaks[cell][::-1]:
			future[:peak + 1, cell] = peak

	# create state vectors
	pstates = past.argsort(axis = 1) / (cell_size - 1)
	fstates = future.argsort(axis = 1) / (cell_size - 1)

	# define metric on the state space
	def gen_metric(dim):
		vec = np.arange(dim) / (dim - 1)
		diff = vec - vec[::-1]
		_max = math.sqrt(np.dot(diff, diff))
		def _metric(dst, src, norm = _max):
			diff = dst - src
			return math.sqrt(np.dot(diff, diff)) / norm
		return _metric
	metric = gen_metric(cell_size)

	# distance
	reconst = np.zeros((time_size, cell_size))
	time_scale = interval / math.log(64)
	for cell in range(cell_size):
		for peak in peaks[cell]:
			portion = reconst[peak:, cell]
			for index in range(len(portion)):
				portion[index] += np.exp(-index / time_scale)
	'''
	plt.imshow(reconst.T, aspect = 'auto')
	activity = np.empty(time_size - 1)
	for time in range(len(activity)):
		activity[time] = metric(reconst[time], reconst[time + 1])
	'''

	print(reconst.shape, time_scale, term, n_components)
	pca_rec = canal.core.pca_exe(reconst, int(time_scale * term), n_components)
	pca_past = canal.core.pca_exe(pstates, int(time_scale * term), n_components)
	return reconst, pca_rec, pca_past
