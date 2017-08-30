import numpy as np
import os
import utils


def load_dataset(dwis, dir_path='./data/gen/', split_ratio=(0.7, 0.2, 0.1)):
	"""
	Load the dataset from disk
	@param dwis: number of diffusion weighted images, i.e features in dataset 
	@param dir_path: string, the super-directory of the data 
	@param split_ratio: how to split the dataset in (training, validation, test) 
	@return: 
	"""

	folders = [folder for folder in os.listdir(dir_path) if folder != '.DS_Store']

	X_list = []
	t_list = []

	# Go trough each folder in the directory
	for folder in folders:
		X_path = dir_path + folder + str('/cylinders.bfloat')
		t_path = dir_path + folder + str('/targets.txt')

		X = utils.to_voxels(utils.read_float(X_path), dwis)
		t = np.loadtxt(t_path, dtype='>f4').reshape(-1, 2)

		X_list.append(X)
		t_list.append(t)

	X = np.concatenate(X_list)
	t = np.concatenate(t_list)
	return _split(X, t[:, 0].reshape(-1, 1), split_ratio)


# Splits dataset into three folds (training, val, test)
def _split(X, t, ratio=(0.7, 0.2, 0.1)):
		no_samples = X.shape[0]
		split = int(no_samples * ratio[0])
		split2 = split + int(no_samples * ratio[1])
		indices = np.random.permutation(no_samples)
		training_idx, valid_idx, test_idx = indices[0:split], indices[split:split2], indices[split2:]
		return (X[training_idx, :], t[training_idx, :]), (X[valid_idx, :], t[valid_idx, :]),  (X[test_idx, :], t[test_idx, :])
