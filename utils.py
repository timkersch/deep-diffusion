from __future__ import division
import numpy as np
import nibabel as nib
from dipy.viz import window, actor


file1 = './data/10000voxels_uniform_p=0_rad=1E-6_sep=2.1E-6_HPC-scheme.bfloat'
file2 = './data/10000voxels_uniform_p=0_rad=1E-6_sep=2.1E-6_HPC-scheme2_seed=843276243.bfloat'
hpcBfloat = '/Users/maq/Documents/School/deep-diffusion/other/data/HPC Subject 100307/T1w/Diffusion/dwi.Bfloat'

hpcfile = '/Users/maq/Documents/School/deep-diffusion/other/data/HPC Subject 100307/T1w/Diffusion/data.nii.gz'


def read_float(filename):
	f = open(filename, "r")
	arr = np.fromfile(f, dtype='>f4')
	return arr


def to_voxels(arr, no_samples=10000, skip_ones=False):
	# N = no samples = 10 000
	# C = channels = 288
	# W = width = 1
	# H = height = 1
	# D = depth = 1
	channels = int(arr.size / no_samples)
	if skip_ones:
		return np.reshape(arr, (no_samples, channels))
	return np.reshape(arr, (no_samples, 1, 1, 1, channels))


def compute_stats():
	f1 = read_float(file1)
	f2 = read_float(file2)

	f1 = to_voxels(f1, skip_ones=True)
	f2 = to_voxels(f2, skip_ones=True)

	return np.abs(np.mean(f1, axis=0) - np.mean(f2, axis=0)), np.abs(np.std(f1, axis=0) - np.std(f2, axis=0))


def read_ni(filename):
	arr = nib.load(filename)
	return arr


def visualize():
	img = nib.load(hpcfile)
	data = img.get_data()
	affine = img.get_affine()
	renderer = window.Renderer()
	# renderer.background((1, 1, 1))

	mean, std = data[data > 0].mean(), data[data > 0].std()
	value_range = (mean - 0.5 * std, mean + 1.5 * std)

	slice_actor = actor.slicer(data, affine, value_range)
	renderer.add(slice_actor)
	slice_actor2 = slice_actor.copy()

	slice_actor2.display(slice_actor2.shape[0]//2, None, None)

	renderer.add(slice_actor2)

	renderer.reset_camera()
	renderer.zoom(1.4)

	window.show(renderer, size=(600, 600), reset_camera=False)

