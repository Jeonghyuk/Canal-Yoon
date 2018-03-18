import numpy as np
import math
import matplotlib.colors as cl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D as ax3d

def scatter(points, labels, colors = None):
    unique_labels = sorted(set(labels))
    if colors is None:
        colors = get_colors(len(unique_labels))
        
    plt.figure()
    for num, label in enumerate(unique_labels):
        plt.scatter(*points[label == labels].T, color = colors[num])
    plt.show()
    
def colormap_to_lut(colormap, with_gamma=False):
    lut = colormap(range(colormap.N))
    if with_gamma:
        return lut
    else:
        return lut[:, :3]
        
def save_lut(filename, lut):
    lut_array = lut if isinstance(lut, np.ndarray) else np.array(lut)
    if lut_array.dtype == float:
        lut_buffer = (255 * lut_array).astype(np.int8)
    elif lut_array.dtype == int:
        lut_buffer = lut_array.astype(np.int8)
    else:
        lut_buffer = lut_array
        
    with open(filename, 'wb') as f:
        for elem in lut_buffer.T.reshape(lut_buffer.size):
            f.write(elem)
    
def orthogonal_view(image, point, **kwargs):
    z, y, x = np.asarray(point).round().astype(int)
    zend, yend, xend = image.shape
    gs = plt.GridSpec(2, 2, hspace=0, wspace=0, width_ratios=(xend, zend),
                      height_ratios=(yend, zend))

    # xy-plane
    ax0 = plt.subplot(gs[0])
    ax0.plot([x] * 2, [0, yend], 'y')
    ax0.plot([0, xend], [y] * 2, 'y')
    ax0.imshow(image[z], **kwargs)
    ax0.xaxis.tick_top()
    ax0.xaxis.set_label_position('top')
    ax0.set_xlabel('X')
    ax0.set_ylabel('Y')

    # yz-plane
    ax1 = plt.subplot(gs[1])#, sharey=ax0)
    ax1.plot([z] * 2, [0, yend], 'y')
    ax1.plot([0, zend], [y] * 2, 'y')
    ax1.imshow(image[..., x].T, **kwargs)
    ax1.yaxis.tick_right()
    ax1.yaxis.set_label_position('right')
    ax1.set_xlabel('Z')
    ax1.set_ylabel('Y')

    # xz-plane
    ax2 = plt.subplot(gs[2])#, sharex=ax0)
    ax2.plot([x] * 2, [0, zend], 'y')
    ax2.plot([0, xend], [z] * 2, 'y')
    ax2.imshow(image[:, y], **kwargs)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Z')

    plt.show()
    
def polar_errorbar(theta, r, style = 'k.', rerr = None, terr = None):
	ax = plt.subplot(111, polar = True)
	ax.plot(theta, r, style)

	res = 100
	if terr != None:
		for err, theta, r in zip(terr, theta, r):
			arc = np.linspace(theta - err, theta + err, res)
			ax.plot(arc, [r] * res, 'k-')

	if rerr != None:
		for err, theta, r in zip(rerr, theta, r):
			bar = [r - err, r + err]
			ax.plot([theta] * 2, bar, 'k-')

def get_colors(num, shuffle = True):
	color_hsv = np.c_[np.linspace(0, 1, num, endpoint = False), np.ones(num), np.ones(num)]
	color_rgb = cl.hsv_to_rgb(color_hsv[np.newaxis, :])[0]

	if shuffle == True:
		color_shuffle = np.empty(color_rgb.shape)
		color_shuffle[::2] = color_rgb[num // 2:, :]
		color_shuffle[1::2] = color_rgb[:num // 2, :]
		return color_shuffle
	else:
		return color_rgb

def show_positions(backgrounds, positions, labels, colors, marker = '+', markersize = 4):
	height_size, row_size, col_size = backgrounds.shape
	drawing_heights = set(positions[:, 0])

	if not isinstance(colors, (tuple, list)):
		colors = (colors,) * len(positions)

	packs = [[] for height in range(height_size)]
	for position, label, color in zip(positions, labels, colors):
		height = position[0]
		packs[height].append((position, label, color))

	for drawing_height in drawing_heights:
		plt.figure()
		plt.title('Z = {} Plane'.format(drawing_height), fontsize = 24)
		plt.xlim((0, col_size - 1))
		plt.ylim((0, row_size - 1))
		#plt.gca().invert_yaxis()
		plt.imshow(backgrounds[drawing_height], cmap = cm.Greys_r, aspect = 'auto')

		for position, label, color in packs[drawing_height]:
			y, x = position[1:]
			plt.plot(x, y, color = 'red', marker = marker, markersize = markersize)
			plt.text(x, y, label, color = color)
		plt.show()

def plot_normalized(normalized, labels, peaks):
	cell_size, time_size = normalized.shape
	cell_colors = get_colors(cell_size, shuffle = False)

	if isinstance(peaks, list):
		wave_size = max([len(elem) for elem in peaks])
		wave_colors = ['black'] * wave_size
	else:
		cell_size, wave_size = peaks.shape
		wave_colors = get_colors(wave_size)

	for cell in range(cell_size):
		plt.plot(normalized[cell], color = cell_colors[cell], label = labels[cell])
		for wave, peak in enumerate(peaks[cell]):
			if not math.isnan(peak):
				plt.plot([peak] * 2, [0, normalized[cell, peak]], linestyle = '--', color = wave_colors[wave])

	plt.xlim((0, time_size))
	plt.xlabel('Time [Frame]')
	plt.ylabel(r'$\Delta F / F$')
	plt.title('Normalized Intensity', fontsize = 24)
	plt.legend()
	plt.show()

def imshow_normalized(normalized, uni = True):
	if uni == True:
		modified_nor = normalized / normalized.max(axis = 1)[:, np.newaxis]
		modified_nor[modified_nor < 0] = 0
		plt.imshow(modified_nor, origin = 'lower', aspect = 'auto')
	else:
		plt.imshow(normalized, origin = 'lower', aspect = 'auto')

	plt.xlabel('Time [Frame]')
	plt.ylabel('Cell Number')
	plt.title('Normalized Intensity', fontsize = 24)
	plt.show()

def plot_raw(raw, labels, peaks = None):
	cell_size, time_size = raw.shape
	cell_colors = get_colors(cell_size, shuffle = False)

	for cell in range(cell_size):
		plt.plot(raw[cell], color = cell_colors[cell], label = labels[cell])

	if peaks != None:
		if isinstance(peaks, list):
			wave_size = max([len(elem) for elem in peaks])
			wave_colors = ['black'] * wave_size
		else:
			cell_size, wave_size = peaks.shape
			wave_colors = get_colors(wave_size)

		for cell in range(cell_size):
			for wave, peak in enumerate(peaks[cell]):
				if peak != -1:
					plt.plot([peak] * 2, [0, raw[cell, peak]], linestyle = '--', color = wave_colors[wave])

	plt.xlim((0, time_size))
	plt.xlabel('Time [Frame]')
	plt.ylabel('$F$ [a.u.]')
	plt.title('Raw Intensity', fontsize = 24)
	plt.legend()
	plt.show()

def imshow_raw(raw, uni = True):
	if uni == True:	
		raw_min = raw.min(axis = 1)
		div = raw.max(axis = 1) - raw_min
		modified_raw = (raw - raw_min[:, np.newaxis]) / div[:, np.newaxis]

		plt.imshow(modified_raw, origin = 'lower', aspect = 'auto', interpolation = 'none')
	else:
		plt.imshow(raw, origin = 'lower', aspect = 'auto', interpolation = 'none')

	plt.xlabel('Time [Frame]')
	plt.ylabel('Cell Number')
	plt.title('Raw Intensity', fontsize = 24)
	plt.show()
'''
def plot_lag(self, cell, marker = '.', markersize = 1):
	if not self._overlap:
		plt.figure()

	if 'lag' in self._dict:
		lag = self._dict['lag']
	else:
		raise Exception('Data does not exist')

	plt.plot(lag[:, cell], marker = marker, linestyle = '', markersize = markersize)
	canal.plot.fit_xrange([0, lag.shape[0]])
	plt.xlabel('Cell Number')
	plt.ylabel('Lag [Frame]')
	plt.title('Time Lag (compared with Cell #{})'.format(cell), fontsize = 24)
	plt.show()
'''
def plot_phases(phases, error):
	cell_size = len(phases)
	plt.errorbar(np.arange(cell_size), phases, error)
	fit_xrange([0, cell_size])
	plt.xlabel('Cell Number')
	plt.ylabel('Phase [a.u.]')
	plt.title('Phases', fontsize = 24)
	plt.show()
'''
def imshow_lag(self):
	if not self._overlap:
		plt.figure()

	if 'lag' in self._dict:
		lag = self._dict['lag']
	else:
		raise Exception('Data does not exist')

	plt.imshow(lag, aspect = 'auto')
	plt.show()
'''
def plot_peaks(peaks, shape, colors = None, monochrome = False, marker = '.', markersize = 1):
	# plot settings
	plt.xlabel('Time [Frame]')
	plt.ylabel('Cell Number')
	plt.title('Peaks', fontsize = 24)

	plt.xlim((-1, shape[1]))
	plt.ylim((-1, shape[0]))
	if isinstance(peaks, list):
		for cell, cell_peaks in enumerate(peaks):
			plt.plot(cell_peaks, [cell] * len(cell_peaks), color = 'black', marker = marker, markersize = markersize, linestyle = '')
	else:
		cell_size, wave_size = peaks.shape
		if colors == None:
			colors = get_colors(wave_size) if monochrome == False else ['black'] * wave_size

		for wave in range(wave_size):
			cells_and_peaks = [[cell, peak] for cell, peak in enumerate(peaks[:, wave]) if not math.isnan(peak)]
			cells, cell_peaks = np.array(cells_and_peaks).T
			plt.plot(cell_peaks, cells, color = colors[wave], marker = marker, markersize = markersize, linestyle = '')
	
	plt.show()

