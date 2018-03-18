import numpy as np
import math
import random
import canal.core
import canal.file
import canal.out
import canal.reconst
import os

import matplotlib.pyplot as plt
import sklearn.manifold
import mpl_toolkits.mplot3d
import scipy.stats

class Canal:
	"""
	Calcium imaging analyzer
	"""

	# for standard output
	_cursor_up = '\x1b[1A'
	_erase_line = '\x1b[2K'

	def __init__(self, json_name = None):
		if json_name == None:
			print('Workflow: load > preprocess > find_peaks > lateral')
			self._dict = {}
		else:
			self._dict = canal.file.load_json(json_name)

	def load(self, folder):
		# get all tiff files in the folder
		def get_info(path):
			lst = [path + elem for elem in os.listdir(path)]
			tifs, txts = [], []
			for elem in lst:
				if elem.endswith('.tif'):
					tifs.append(elem)
				elif elem.endswith('.txt'):
					txts.append(elem)

			with open(txts[0], 'r') as f:
				prefix = 'Z : '
				for line in f:
					if line.startswith(prefix):
						height = int(line[len(prefix):])
						return tifs, height
				else:
					raise Exception('No infomation')

		# get folders
		references_folder = folder
		for elem in os.listdir(folder):
			full_path = folder + elem
			if os.path.isdir(full_path):
				signals_folder = full_path + '/'
				break
		else:
			raise Exception('Ill format: No calcium movie folder')

		references = canal.file.loadtif(*get_info(references_folder))
		self._dict['references'] = references
		signals = canal.file.loadtif(*get_info(signals_folder))	
		self._dict['signals'] = signals

	def preprocess(self, radius = (1.5, 5.5, 5.5)):
		# execute registration and splice them
		references = self._dict['references']
		signals = self._dict['signals']
		#aligned_ref, aligned_sig = canal.reconst.splice(references, signals)
		aligned_ref, aligned_sig = references, signals
		self._dict['nucleus'] = aligned_ref.mean(axis = 0)
		self._dict['calcium'] = aligned_sig.mean(axis = 0)

		# get the masks of the cells in the movie
		print('Executing cell segmentation...', end = '\r')
		masks = canal.reconst.masks(aligned_ref, radius)
		print(self._erase_line + '{} cells found'.format(len(masks)))
		self._dict['masks'] = masks

		# calc fluorescence level of the cells
		fluos, positions = canal.reconst.signals(aligned_sig, masks, aligned_ref.shape[1])
		self._dict['dimension'] = aligned_sig.shape

		sorted_joined = np.array(sorted(np.c_[fluos, positions], key = lambda i: i[-1]))
		self._dict['fluos'] = sorted_joined[:, :-3]
		self._dict['positions'] = sorted_joined[:, -3:].astype(int)

	def remove_invalid(self, valid_cells):
		keys1d = ('positions', 'fluos', 'normalized')
		for key in keys1d:
			if key in self._dict:
				prev = self._dict[key]
				self._dict[key] = prev[valid_cells]

	def imshow_raw(self):
		fluos = self._dict['fluos']
		canal.out.imshow_raw(fluos)

	def imshow_normalized(self):
		if 'normalized' in self._dict:
			normalized = self._dict['normalized']
		else:
			raise Exception('Find peaks first')

		canal.out.imshow_normalized(normalized)

	def plot_raw(self, cells = None, colors = 'blue', image = True):
		if isinstance(cells, int):
			cells = [cells]
		elif cells == None:
			cells = range(len(self._dict['fluos']))

		raw = self._dict['fluos'][cells]
		names = self._dict['names']
		labels = [names[cell] for cell in cells]
		# for manipulation
		for cell, name in zip(cells, labels):
			print(str(cell) + ': ' + name)

		if 'peaks' in self._dict:
			peaks = self._dict['peaks']
			canal.out.plot_raw(raw, labels, [peaks[elem] for elem in cells])
		else:
			canal.out.plot_raw(raw, labels)

		if image == True:
			positions = self._dict['positions'][cells]
			backgrounds = self._dict['calcium']
			canal.out.show_positions(backgrounds, positions, labels, colors)

	def plot_peaks(self, activity = None, markersize = 1):
		shape = self._dict['fluos'].shape

		if activity == None:
			if 'peaks' in self._dict:
				peaks = self._dict['peaks']
			else:
				raise Exception('Find peaks first')

			if isinstance(peaks, list):
				canal.out.plot_peaks(peaks, shape, markersize = markersize)
			else:
				if 'wave_profiles' in self._dict:
					wave_color_template = ('yellow', 'red', 'green')
					wave_colors = np.array([wave_color_template[profile + 1] for profile in self._dict['wave_profiles']], dtype = str)
				canal.out.plot_peaks(peaks, shape, colors = wave_colors, markersize = markersize)
		else:
			if 'peaks_grouped' in self._dict:
				peaks_grouped = self._dict['peaks_grouped']
			else:
				raise Exception('Find waves first')

			canal.out.plot_peaks(peaks_grouped[activity], shape, markersize = markersize)
	'''
	def plot_phases(self, activity):
		try:
			phases = self._dict['phases']
			phases__err = self._dict['phases__err']
		except:
			raise Exception('Find waves first')

		canal.out.plot_phases(phases[activity], phases__err[activity])
	'''

	def isomap(self, width, n_neighbors = 30, power = 2, n_components = 3):
		peaks = self._dict['peaks']
		cell_size, time_size = self._dict['fluos'].shape

		if 'distances' not in self._dict:
			# states
			states = np.empty((cell_size, time_size))
			for time in range(time_size):
				print('scanning: time {}'.format(time), end = '\r')
				for cell in range(cell_size):
					cell_peaks = peaks[cell]
					if len(cell_peaks) > 0:
						diffs = np.array([peak - time for peak in cell_peaks])
						#states[cell, time] = cell_peaks[np.argmin(diffs * diffs)]
						states[cell, time] = max(min(diffs[np.argmin(diffs * diffs)], width), -width)
					else:
						states[cell, time] = 0
			self._dict['states'] = states

			# calc distances
			distances = np.zeros([time_size] * 2)
			for dst_time in range(time_size):
				print('scanning: time {}'.format(dst_time), end = '\r')
				for src_time in range(dst_time + 1, time_size):
					distance = np.sum((states[:, dst_time] - states[:, src_time]) ** power)
					distances[dst_time, src_time] = distance
					distances[src_time, dst_time] = distance
			self._dict['distances'] = distances
		else:
			distances = self._dict['distances']
			states = self._dict['states']

		#plt.imshow(distances)

		# create the kernel for kernel PCA
		def centralize(kernel, n_components = n_components):
			return kernel
			factor = np.empty(kernel.shape)
			factor.fill(1 / n_components)
			return kernel - np.dot(factor, kernel) - np.dot(kernel, factor) + np.dot(np.dot(factor, kernel), factor)
		
		# execute kernel PCA
		kernel = centralize(distances)
		pca_solver = sklearn.decomposition.PCA(n_components = n_components).fit(kernel)
		from_dist = pca_solver.transform(kernel).T
		print(from_dist.shape)		

		# test raw PCA
		pca_solver2 = sklearn.decomposition.PCA(n_components = n_components).fit(states.T)
		from_states = pca_solver2.transform(states.T).T
		print(from_states.shape)

		# test isomap
		solver = sklearn.manifold.Isomap(n_neighbors = n_neighbors, n_components = 3)
		solver.fit(states.T)
		isomap = solver.transform(states.T).T
		print(isomap.shape)
		
		fig = plt.figure()
		ax = fig.add_subplot(111, projection = '3d')
		ax.plot(from_dist[0], from_dist[1], from_dist[2])
		plt.show()

		fig = plt.figure()
		ax = fig.add_subplot(111, projection = '3d')
		ax.plot(from_states[0], from_states[1], from_states[2])
		plt.show()
		
		fig = plt.figure()
		ax = fig.add_subplot(111, projection = '3d')
		ax.plot(isomap[0], isomap[1], isomap[2])
		plt.show()

		self._dict['isomap'] = (from_dist, from_states, isomap)

	def analyze(self, term, n_components = 3, wave = False):
		fluos = self._dict['fluos']
		peaks, normalized = canal.core.find_peaks(fluos)

		self._dict['normalized'] = normalized
		self._dict['peaks'] = peaks
		if wave == True:
			peaks, wave_profiles = canal.core.find_waves(peaks, normalized)
			#self.remove_invalid(valid_cells)
			self._dict['peaks'] = peaks
			self._dict['wave_profiles'] = wave_profiles
			peaks_grouped = [[] for count in range(2)]
			for profile in range(2):
				peaks_grouped[profile] = peaks[:, wave_profiles == profile]
			self._dict['peaks_grouped'] = peaks_grouped
		else:
			return

		cell_size, time_size = normalized.shape
		def get_interval(peaks):
			diff_medians = []
			for cell_peaks in peaks:
				if len(cell_peaks) > 1:
					vec = np.array(cell_peaks)
					diff = vec[1:] - vec[:-1]
					diff_medians.append(np.median(diff))
			return np.min(diff_medians)

		interval = get_interval(peaks)
		reconstructed, pca_rec, pca_past = canal.core.find_trajectory(peaks, time_size, interval, term, n_components)
		self._dict['reconstructed'] = reconstructed
		self._dict['pca_rec'] = pca_rec
		self._dict['pca_past'] = pca_past

	def plot_trajectory(self, view_range = None, n_components = 3):
		reconst = self._dict['reconstructed']
		pca_rec = self._dict['pca_rec']
		pca_past = self._dict['pca_past']

		if view_range == None:
			peaks_grouped = self._dict['peaks_grouped']
			view_range = []
			for group in peaks_grouped:
				group_view = []
				for peaks in group.T:
					group_view.append((peaks.min(), peaks.max()))
				view_range.append(group_view)

		canal.core.pca_view(pca_rec, n_components, view_range)
		canal.core.pca_view(pca_past, n_components, view_range)

	def plot_phases(self, activity):
		phases, phases_err = canal.core.calc_phases(self._dict['peaks_grouped'][activity])
		canal.out.plot_phases(phases, phases_err)

	def identity(self, cells, candidates):
		peaks_grouped = self._dict['peaks_grouped']
		forward_phases = peaks_grouped[0]
		backward_phases = peaks_grouped[1]

		output = []
		for cell in cells:
			forward_diffs = forward_phases[cell] - forward_phases[candidates]
			backward_diffs = backward_phases[cell] - backward_phases[candidates]

			forward_score = np.sqrt(forward_diffs * forward_diffs)
			forward_score = np.mean(forward_score / forward_score.max(axis = 0), axis = 1)
			backward_score = np.sqrt(backward_diffs * backward_diffs)
			backward_score = np.mean(backward_score / backward_score.max(axis = 0), axis = 1)

			output.append((candidates[forward_score.argmin()], candidates[backward_score.argmin()]))
		return output

	def lateral(self, view_range = None):
		peaks_grouped = self._dict['peaks_grouped']
		if view_range == None:
			view_range = (0, len(peaks_grouped[0]))
		forward_phases = peaks_grouped[0][view_range[0]:view_range[1]]
		backward_phases = peaks_grouped[1][view_range[0]:view_range[1]]

		positions = self._dict['positions']
		ana_thres = self._dict['dimension'][-1] / 6
		cell_size = len(forward_phases)
		lateral_cells = []
		ttest_thres = 0.01
		for src_cell in range(cell_size):
			for dst_cell in range(src_cell):
				forward_diffs = forward_phases[dst_cell, :] - forward_phases[src_cell, :]
				backward_diffs = backward_phases[dst_cell, :] - backward_phases[src_cell, :]
				forward_stat, forward_prob = scipy.stats.ttest_1samp(forward_diffs, 0)
				backward_stat, backward_prob = scipy.stats.ttest_1samp(backward_diffs, 0)
				if forward_prob < ttest_thres and backward_prob < ttest_thres and abs(positions[dst_cell, -1] - positions[src_cell, -1]) < ana_thres:
					if forward_stat > 0 and backward_stat > 0:
						lateral_cells.append(dst_cell)
					elif forward_stat < 0 and backward_stat < 0:
						lateral_cells.append(src_cell)

		hist, edges = np.histogram(lateral_cells, bins = range(max(lateral_cells)))

		packed = list(set(lateral_cells))
		plt.figure()
		plt.plot(positions[packed, -1], positions[packed, -2], 'o')
		for cell in packed:
			try:
				plt.text(positions[cell, -1], positions[cell, -2], str(hist[cell]))
			except:
				pass

		backgrounds = self._dict['calcium']
		def str_list(lst):
			return (str(elem) for elem in lst)
		canal.out.show_positions(backgrounds, positions[packed], str_list(list(set(lateral_cells))))
		return lateral_cells

class Canal_legacy:
	"""
	Calcium imaging analyzer
	"""

	# for standard output
	_cursor_up = '\x1b[1A'
	_erase_line = '\x1b[2K'

	def __init__(self, load_string, debug = False):
		self._file_name = None	# filename to load
		self._overlap = False
		self.load(load_string)

		self._debug = debug

	def load(self, load_string = None):
		if load_string == None:
			if self._file_name != None:
				self._dict = canal.file.load(self._file_name)
			else:
				raise Exception('Enter a file name or folder')
		else:
			self._dict = canal.file.load(load_string)
			self._file_name = load_string

	def save(self, file_name = None):
		if file_name == None:
			if self._file_name != None:
				file_name = canal.file.pickle_name(self._file_name)
				canal.file.save_pickle(file_name, self._dict)
				self._file_name = file_name
			else:
				raise Exception('Enter a file name')
		else:
			file_name = canal.file.pickle_name(file_name)
			canal.file.save_pickle(file_name, self._dict)
			self._file_name = file_name

	def overlap(self, flag = True):
		if flag == True or flag == False:
			self._overlap = flag
		else:
			raise Exception('True or False')

	def analyze(self):
		self.normalize()
		self.find_peaks()
		self.find_waves()

		self.plot_peaks(0, markersize = 4)

	def add_group(self, group, *indices):
		groups = self._dict['groups']
		groups[group] = indices

	def pop_group(self, group):
		groups = self._dict['groups']
		groups.pop(elem)

	def add_label(self, label, index):
		labels = self._dict['labels']
		labels[index] = label

	def pop_group(self, label):
		labels = self._dict['labels']
		for index, _label in labels.items():
			if label == _label:
				labels.pop(index)

	def indices(self, *strings):
		def _indices(labels, groups, string):
			for index, label in labels.items():
				if string == label:
					return [index]

			strings = string.split(' ')
			sets = []
			for group in strings:
				if group in groups:
					sets.append(set(groups[group]))

			result = sets[0]
			for elem in sets[1:]:
				result.intersection_update(elem)
			return result

		labels = self._dict['labels']
		groups = self._dict['groups']
		union = set()
		for string in strings:
			result = _indices(labels, groups, string)
			union = union.union(result)

		return sorted(union)
		
	def remove_invalid(self, valid_cells):
		keys1d = ('position', 'raw', 'normalized')
		for key in keys1d:
			if key in self._dict:
				prev = self._dict[key]
				self._dict[key] = prev[valid_cells]

		convertor = {}
		new_index = 0
		for old_index in valid_cells:
			convertor[old_index] = new_index
			new_index += 1

		if 'labels' in self._dict:
			labels = self._dict['labels']
			new_labels = {}
			for index, label in labels.items():
				try:
					new_labels[convertor[index]] = label
				except KeyError:
					pass
			self._dict['labels'] = new_labels

		if 'groups' in self._dict:
			groups = self._dict['groups']
			new_groups = {}
			for group, indices in groups.items():
				new_indices = []
				for index in indices:
					try:
						new_indices.append(convertor[index])
					except KeyError:
						pass

				if len(new_indices) != 0:
					new_groups[group] = new_indices
			self._dict['groups'] = new_groups

	def select(self, *groups):
		self._dict['selected'] = groups

	def imshow_raw(self):
		raw = self._dict['raw']
		canal.out.imshow_raw(raw)

	def plot_raw(self):
		try:
			selected = self._dict['selected']
		except:
			raise Exception('Select groups first')

		cells = self.indices(*selected)
		raw = self._dict['raw'][cells]
		_labels = self._dict['labels']
		labels = [_labels.get(cell, str(cell)) for cell in cells]
		if 'peaks' in self._dict:
			peaks = self._dict['peaks']
			canal.out.plot_raw(raw, labels, peaks)
		else:
			canal.out.plot_raw(raw, labels)

		positions = self._dict['position'][cells]
		backgrounds = self._dict['movie_stacked']
		canal.out.show_positions(backgrounds, positions, labels)

	def imshow_normalized(self):
		if 'normalized' in self._dict:
			normalized = self._dict['normalized']
		else:
			raise Exception('Find peaks first')

		canal.out.imshow_normalized(normalized)

	def plot_normalized(self):
		try:
			selected = self._dict['selected']
		except:
			raise Exception('Select cells first')

		try:
			peaks = self._dict['peaks']
			normalized = self._dict['normalized']
		except:
			raise Exception('Find peaks first')

		cells = self.indices(*selected)
		normalized = self._dict['normalized'][cells]
		_labels = self._dict['labels']
		labels = [_labels.get(cell, str(cell)) for cell in cells]
		canal.out.plot_normalized(normalized, labels, peaks)

		positions = self._dict['position'][cells]
		backgrounds = self._dict['movie_stacked']
		canal.out.show_positions(backgrounds, positions, labels)

	def plot_peaks(self, activity = None, markersize = 1):
		shape = self._dict['raw'].shape
		if activity == None:
			if 'peaks' in self._dict:
				peaks = self._dict['peaks']
			else:
				raise Exception('Find peaks first')

			if 'wave_profiles' in self._dict:
				wave_color_template = ('yellow', 'red', 'green')
				wave_colors = np.array([wave_color_template[profile + 1] for profile in self._dict['wave_profiles']], dtype = str)
			canal.out.plot_peaks(peaks, shape, colors = wave_colors, markersize = markersize)
		else:
			if 'peaks_grouped' in self._dict:
				peaks_grouped = self._dict['peaks_grouped']
			else:
				raise Exception('Find waves first')

			canal.out.plot_peaks(peaks_grouped[activity], shape, markersize = markersize)
	'''
	def plot_phases(self, activity):
		try:
			phases = self._dict['phases']
			phases__err = self._dict['phases__err']
		except:
			raise Exception('Find waves first')

		canal.out.plot_phases(phases[activity], phases__err[activity])
	'''
	def normalize(self):
		raw = self._dict['raw']
		normalized, filter_widths, cycles, valid_cells = canal.core.normalize(raw, detailed = self._debug)

		self.remove_invalid(valid_cells)
		self._dict['normalized'] = normalized
		self._dict['filter_widths'] = filter_widths
		self._dict['cycles'] = cycles

	def find_peaks_legacy(self):
		try:
			normalized = self._dict['normalized']
			filter_widths = self._dict['filter_widths']
			cycles = self._dict['cycles']
		except:
			raise Exception('Normalize first')

		peaks, valid_cells = canal.core.find_peaks(normalized, filter_widths, cycles)

		self.remove_invalid(valid_cells)
		self._dict['peaks'] = peaks

	def find_peaks(self):
		raw = self._dict['raw']

		peaks, valid_cells, wave_profiles = canal.core.find_peaks(raw)

		self.remove_invalid(valid_cells)
		self._dict['peaks'] = peaks
		self._dict['wave_profiles'] = wave_profiles
		peaks_grouped = [[] for count in range(2)]
		for profile in range(2):
			peaks_grouped[profile] = peaks[:, wave_profiles == profile]
		self._dict['peaks_grouped'] = peaks_grouped

	def plot_phases(self, activity):
		phases, phases_err = canal.core.calc_phases(self._dict['peaks_grouped'][activity])
		canal.out.plot_phases(phases, phases_err)

	def lateral(self):
		peaks_grouped = self._dict['peaks_grouped']
		forward_phases = peaks_grouped[0]
		backward_phases = peaks_grouped[1]

		position = self._dict['position']
		ana_thres = self._dict['dimension'][-1] / 6
		cell_size = len(forward_phases)
		lateral_cells = []
		ttest_thres = 0.01
		for src_cell in range(cell_size):
			for dst_cell in range(src_cell):
				forward_dst, forward_src = forward_phases[dst_cell, :], forward_phases[src_cell, :]
				forward_diffs = [dst - src for dst, src in zip(forward_dst, forward_src) if dst != -1 and src != -1]
				backward_dst, backward_src = backward_phases[dst_cell, :], backward_phases[src_cell, :]
				backward_diffs = [dst - src for dst, src in zip(backward_dst, backward_src) if dst != -1 and src != -1]
				if len(forward_diffs) == 0 or len(backward_diffs) == 0:
					continue
				forward_stat, forward_prob = scipy.stats.ttest_1samp(forward_diffs, 0)
				backward_stat, backward_prob = scipy.stats.ttest_1samp(backward_diffs, 0)
				if forward_prob < ttest_thres and backward_prob < ttest_thres and abs(position[dst_cell, -1] - position[src_cell, -1]) < ana_thres:
					if forward_stat > 0 and backward_stat > 0:
						lateral_cells.append(dst_cell)
					elif forward_stat < 0 and backward_stat < 0:
						lateral_cells.append(src_cell)

		hist, edges = np.histogram(lateral_cells, bins = range(max(lateral_cells)))

		packed = list(set(lateral_cells))
		plt.figure()
		plt.plot(position[packed, -1], position[packed, -2], 'o')
		for cell in packed:
			try:
				plt.text(position[cell, -1], position[cell, -2], str(hist[cell]))
			except:
				pass

		backgrounds = self._dict['movie_stacked']
		canal.out.show_positions(backgrounds, position[packed])
		return lateral_cells
	'''
	def find_distinct_pairs(self, distance, reject = 0.05):
		import scipy.stats
		peaks = self._dict['peak_individual']
		positions = self._dict['position']
		cell_size = len(positions)

		peak_index_sync = self._dict['peak_index_sync']
		ttest = np.zeros([cell_size] * 2, dtype = bool)
		for dst_cell in range(cell_size):
			print('Comparing: Cell #{} and the others'.format(dst_cell), end = '\r')
			for src_cell in range(dst_cell + 1, cell_size):
				dst_peaks = np.array(peaks[dst_cell])[peak_index_sync[dst_cell][src_cell]]
				src_peaks = np.array(peaks[src_cell])[peak_index_sync[src_cell][dst_cell]]
				pos_diff = positions[src_cell][1:] - positions[dst_cell][1:]
				dist = (pos_diff * pos_diff).sum()
				if dist < distance and len(dst_peaks) != 0 and len(src_peaks) != 0:
					statistic, prob = scipy.stats.ttest_1samp(dst_peaks - src_peaks, 0)
					if prob < reject:
						ttest[dst_cell, src_cell], ttest[src_cell, dst_cell] = [True] * 2

		print(Canal._erase_line, end = '')
		self._dict['ttest'] = ttest

		distinct = [[cell, elem.astype(int).sum()] for cell, elem in enumerate(ttest)]
		self.imshow_position([elem[0] for elem in sorted(distinct, key = lambda i: i[-1])[-30:]])

	def isomap(self, uni = True):
		normalized = self._dict['normalized']
		denoising_scale = self._dict['denoising_scale']
		if uni == True:	
			nor_min = normalized.min(axis = 1)
			div = normalized.max(axis = 1) - nor_min
			normalized = (normalized - nor_min[:, np.newaxis]) / div[:, np.newaxis]

		solver = sklearn.manifold.Isomap(n_neighbors = 2 * denoising_scale + 1, n_components = 3)
		solver.fit(normalized.T)
		transformed = solver.transform(normalized.T).T

		fig = plt.figure()
		ax = fig.add_subplot(111, projection = '3d')
		ax.plot(transformed[0], transformed[1], transformed[2])
		plt.show()
		self._dict['isomap'] = transformed

	def find_waves(self, short = True, fluc_scale = None):
		if 'peaks' in self._dict:
			peaks = self._dict['peaks']
		else:
			raise Exception('Find peaks first')

		filter_widths = self._dict['filter_widths']
		cycles = self._dict['cycles']

		peaks_grouped, phase, phase__err = canal.core.find_waves(peaks, filter_widths, cycles)

		self._dict['peaks_grouped'] = peaks_grouped
		self._dict['phases'] = phase
		self._dict['phases__err'] = phase__err
	'''
