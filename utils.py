from __future__ import division
import numpy as np
import nibabel as nib

file1 = './data/10000voxels_uniform_p=0_rad=1E-6_sep=2.1E-6_HPC-scheme.bfloat'
file2 = './data/10000voxels_uniform_p=0_rad=1E-6_sep=2.1E-6_HPC-scheme2_seed=843276243.bfloat'
hpc = '/Users/maq/Documents/School/deep-diffusion/other/data/HPC Subject 100307/T1w/Diffusion/dwi.Bfloat'

fileList = ['./data/1000voxels_uniform_p=0_rad=0.1E-6_sep=1.1E-6_HPC-scheme.bfloat',
			'./data/1000voxels_uniform_p=0_rad=0.1E-6_sep=2.1E-6_HPC-scheme.bfloat',
			'./data/1000voxels_uniform_p=0_rad=0.5E-6_sep=1.1E-6_HPC-scheme.bfloat',
			'./data/1000voxels_uniform_p=0_rad=1.5E-6_sep=3.1E-6_HPC-scheme.bfloat',
			'./data/1000voxels_uniform_p=0_rad=1E-6_sep=3.1E-6_HPC-scheme.bfloat',
			'./data/1000voxels_uniform_p=0_rad=2E-6_sep=4.1E-6_HPC-scheme.bfloat']

targetList = [(0.1E-6, 1.1E-6),
			  (0.1E-6, 2.1E-6),
			  (0.5E-6, 1.1E-6),
			  (1.5E-6, 3.1E-6),
			  (1E-6, 3.1E-6),
			  (2E-6, 4.1E-6)]


def read_float(filename):
	f = open(filename, "r")
	arr = np.fromfile(f, dtype='>f4')
	return arr


def read_floats(filenames):
	files = []
	for file in range(0, len(filenames)):
		files.append(to_voxels(read_float(filenames[file])))
	return files


def read_ni(filename):
	arr = nib.load(filename)
	return arr


def to_voxels(arr, channels=288, skip_ones=True):
	# N = no samples = 1000
	# C = channels = 288
	# W = width = 1
	# H = height = 1
	# D = depth = 1
	no_samples = int(arr.size / channels)
	if skip_ones:
		return np.reshape(arr, (no_samples, channels))
	return np.reshape(arr, (no_samples, 1, 1, 1, channels))


def get_pred_data(sample_size=None):
	arr = read_float(hpc)
	arr = arr.reshape(arr.shape[0]/288, 288)
	if sample_size is not None:
		arr = np.random.shuffle(arr)
		return arr[0:sample_size, :]
	return arr


def load_data():
	X = np.empty((len(fileList) * 1000, 288))
	y = np.empty((len(fileList) * 1000, 2))

	start = 0
	end = 1000
	for i in xrange(0, len(fileList)):
		file = fileList[i]
		targetTuple = targetList[i]
		vals = to_voxels(read_float(file), skip_ones=True)

		X[start:end] = vals
		y[start:end] = np.array(targetTuple)

		start = end
		end = end + 1000

	return X, y


def get_data(split_ratio=0.7):
	X, y = load_data()
	split = int(X.shape[0] * split_ratio)
	indices = np.random.permutation(X.shape[0])
	training_idx, test_idx = indices[:split], indices[split:]
	return X[training_idx, :], y[training_idx, :], X[test_idx, :], y[test_idx, :]
