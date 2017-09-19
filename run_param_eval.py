from matplotlib.ticker import FormatStrFormatter

import utils
from sklearn.neighbors import KNeighborsRegressor
import numpy as np
from matplotlib.pyplot import hist
import matplotlib.pyplot as plt


def knn(input, targets, hpc):
	"""
	Method used for fitting kNN to training data
	Used for comparing HPC data with generated data
	@param X_train: The input training data
	@param y_train: The target training data
	@param X_val: The input validation data
	@param y_val: The target validation data
	@param X_hpc: The HPC input data
	@param class_index: The class to fit on (0 = cylinder radius, 1 = cylinder separation)
	@return: nothing 
	"""
	# Fit model to training data
	model = KNeighborsRegressor(n_neighbors=10, weights='distance', algorithm='auto', metric='euclidean', n_jobs=4)
	model.fit(input, targets)

	# Print kNN scores for train and validation set to measure how good the fit is
	print('Score train: ' + str(model.score(input, targets)))

	# Make prediction on the HPC set
	predictions = model.predict(hpc)
	#print(predictions[np.where((predictions >= 1e-10) & (predictions <= 5e-10))].shape)
	#print(predictions[np.where((predictions <= 1e-9) & (predictions > 5e-10))].shape)
	print('Predictions:')
	print('Min: ' + str(np.min(predictions)))
	print('Max: ' + str(np.max(predictions)))

	print('')

	print('Targets:')
	print('Min: ' + str(np.min(targets)))
	print('Max: ' + str(np.max(targets)))

	predictions = predictions[np.where(predictions <= 1e-11)]

	fig, ax = plt.subplots()
	ax.xaxis.set_major_formatter(FormatStrFormatter('%.1e'))
	n, bins, patches = hist(predictions, rwidth=0.8, bins=20, facecolor='green')
	print('BINS: %i', bins)
	print('N: %i', n)



	plt.xlabel('Cylinder radius')
	plt.ylabel('Instance count')
	plt.title('Histogram of cylinder radiuses')
	plt.xticks(bins)
	plt.setp(ax.get_xticklabels(), rotation=90, horizontalalignment='right')
	plt.grid(True)

	plt.show()

if __name__ == '__main__':
	# Load the generated dataset
	X_train, y_train, _, _ = utils.get_param_eval_data(split_ratio=1.0)
	print(X_train.shape)
	print(y_train.shape)

	# Load randomly sampled HPC voxels
	X_hpc = utils.get_hpc_data(sample_size=50000)
	X_hpc = utils.filter_zeros(X_hpc)

	# Run on cylinder radius, i.e class index 0
	knn(X_train, y_train, X_hpc)
