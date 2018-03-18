import numpy as np
import scipy.ndimage.filters
import scipy.ndimage.morphology
import math
import canal.opt
import canal.local
import random
import sklearn.decomposition
import cv2
import itertools

import matplotlib.pyplot as plt

_cursor_up = '\x1b[1A'
_erase_line = '\x1b[2K'
_plusminus = u'\u00B1'

def pca_exe(data, ext, n_components):
	# execute PCA
	_extended = extend(data, ext)
	pca_solver = sklearn.decomposition.PCA(n_components = n_components).fit(_extended)
	return pca_solver.transform(_extended)

def pca_view(data, n_components, line_range = [], view_range = []):
	if view_range == []:
		view_range = [[(0, len(data))]]
	colors = canal.out.get_colors(len(view_range), shuffle = False)
	# show results (temp)
	if n_components == 3:
		from mpl_toolkits.mplot3d import Axes3D
		fig = plt.figure()
		ax = Axes3D(fig)
		for group, views in enumerate(view_range):
			for view in views:
				ax.scatter(data[view[0]:view[1], 0], data[view[0]:view[1], 1], data[view[0]:view[1], 2], color = colors[group])
			ax.plot(data[:, 0][line_range[0]:line_range[1]], data[:, 1][line_range[0]:line_range[1]], data[:, 2][line_range[0]:line_range[1]], color = 'black')
	elif n_components == 2:
		from mpl_toolkits.mplot3d import Axes3D
		fig = plt.figure()
		ax = Axes3D(fig)
		for group, views in enumerate(view_range):
			for view in views:
				ax.scatter(data[view[0]:view[1], 0], data[view[0]:view[1], 1], np.arange(view[0], view[1]), color = colors[group])
		ax.plot(data[:, 0][line_range[0]:line_range[1]], data[:, 1][line_range[0]:line_range[1]], np.arange(line_range[0], line_range[1]), color = 'black')
		plt.figure()
		plt.plot(data[:, 0][line_range[0]:line_range[1]], data[:, 1][line_range[0]:line_range[1]], color = 'black')

		for group, views in enumerate(view_range):
			for view in views:
				plt.scatter(data[view[0]:view[1], 0], data[view[0]:view[1], 1], color = colors[group])

	elif n_components == 1:
		plt.figure()
		plt.scatter(np.arange(len(data)), data, color = colors)
