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
	@param input: The input training data
	@param targets: The target training data
	@param hpc: The HPC input data to make predictions on
	@return: nothing 
	"""
	# Fit model to training data
	model = KNeighborsRegressor(n_neighbors=10, weights='distance', algorithm='auto', metric='euclidean', n_jobs=4)
	model.fit(input, targets)

	# Make prediction on the HPC set
	predictions = model.predict(hpc)
	predictions = np.log10(predictions)

	fig, ax = plt.subplots()
	ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
	unique_values, counts = np.unique(np.log10(targets), return_counts=True)
	ax.scatter(unique_values, counts, edgecolors=(0, 0, 0), label='Targets')
	n, bins, patches = ax.hist(predictions, rwidth=0.8, bins=30, range=(np.log10(np.min(targets)), np.log10(np.max(targets))), facecolor='green', label='Predictions')

	plt.xlabel('Log base 10 Cylinder radius')
	plt.ylabel('Instance count')
	plt.title('Histogram of log Cylinder radiuses')
	plt.xticks(bins)
	plt.setp(ax.get_xticklabels(), rotation=90, horizontalalignment='right')
	plt.grid(True)
	ax.legend()

	print('BINS: %i', bins)
	print('N: %i', n)

	plt.show()

if __name__ == '__main__':
	# Load the generated dataset
	X_train, y_train, _, _ = utils.get_param_eval_data(split_ratio=1.0)

	# Load randomly sampled HPC voxels
	X_hpc = utils.get_hpc_data(sample_size=50000)
	X_hpc = utils.filter_zeros(X_hpc)

	knn(X_train, y_train, X_hpc)
