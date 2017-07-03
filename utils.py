from __future__ import division
import numpy as np
import nibabel as nib
from matplotlib.pyplot import hist
import matplotlib.pyplot as plt

hpc = './data/dwi.Bfloat'

fileList1 = ['./data/1000voxels_uniform_p=0_rad=0.1E-6_sep=1.1E-6_HPC-scheme.bfloat',
            './data/1000voxels_uniform_p=0_rad=0.1E-6_sep=2.1E-6_HPC-scheme.bfloat',
            './data/1000voxels_uniform_p=0_rad=0.5E-6_sep=1.1E-6_HPC-scheme.bfloat',
            './data/1000voxels_uniform_p=0_rad=1.5E-6_sep=3.1E-6_HPC-scheme.bfloat',
            './data/1000voxels_uniform_p=0_rad=1E-6_sep=3.1E-6_HPC-scheme.bfloat',
            './data/1000voxels_uniform_p=0_rad=2E-6_sep=4.1E-6_HPC-scheme.bfloat']

fileList2 = ['./data/finer/1000voxels_uniform_p=0_rad=1.5E-7_sep=1.1E-6_HPC-scheme.bfloat',
             './data/finer/1000voxels_uniform_p=0_rad=1E-7_sep=1.1E-6_HPC-scheme.bfloat',
             './data/finer/1000voxels_uniform_p=0_rad=1E-8_sep=1.1E-6_HPC-scheme.bfloat',
             './data/finer/1000voxels_uniform_p=0_rad=2E-7_sep=1.1E-6_HPC-scheme.bfloat',
             './data/finer/1000voxels_uniform_p=0_rad=5E-8_sep=1.1E-6_HPC-scheme.bfloat']

files = fileList1 + fileList2

targetList1 = [(0.1E-6, 1.1E-6),
               (0.1E-6, 2.1E-6),
               (0.5E-6, 1.1E-6),
               (1.5E-6, 3.1E-6),
               (1E-6, 3.1E-6),
               (2E-6, 4.1E-6)]

targetList2 = [(1.5E-7, 1.1E-6),
               (1E-7, 1.1E-6),
               (1E-8, 1.1E-6),
               (2E-7, 1.1E-6),
               (5E-8, 1.1E-6)]

targets = targetList1 + targetList2


def get_data(split_ratio=0.7):
	X, y = _load_data(files, targets)
	split = int(X.shape[0] * split_ratio)
	indices = np.random.permutation(X.shape[0])
	training_idx, test_idx = indices[:split], indices[split:]
	return X[training_idx, :], y[training_idx, :], X[test_idx, :], y[test_idx, :]


def get_pred_data(sample_size=None):
	arr = _to_voxels(_read_float(hpc))
	if sample_size is not None:
		np.random.shuffle(arr)
		return arr[0:sample_size, :]
	return arr


def plot_features():
	X, y = _load_data(fileList2)
	for i in range(0, X.shape[1]):
		x = X[:, i]
		n, bins, patches = hist(x, bins='auto', range=None, normed=False, weights=None, cumulative=False, bottom=None)
		plt.show()


def _load_data(file_list, target_list):
	X = np.empty((len(file_list) * 1000, 288))
	y = np.empty((len(file_list) * 1000, 2))

	start = 0
	end = 1000
	for i in xrange(0, len(file_list)):
		file = file_list[i]
		target_tuple = target_list[i]
		vals = _to_voxels(_read_float(file), skip_ones=True)

		X[start:end] = vals
		y[start:end] = np.array(target_tuple)

		start = end
		end = end + 1000

	return X, y


def _read_float(filename):
	f = open(filename, "r")
	arr = np.fromfile(f, dtype='>f4')
	return arr


def _read_ni(filename):
	arr = nib.load(filename)
	return arr


def _to_voxels(arr, channels=288, skip_ones=True):
	# N = no samples = 1000
	# C = channels = 288
	# W = width = 1
	# H = height = 1
	# D = depth = 1
	no_samples = int(arr.size / channels)
	if skip_ones:
		return np.reshape(arr, (no_samples, channels))
	return np.reshape(arr, (no_samples, 1, 1, 1, channels))
