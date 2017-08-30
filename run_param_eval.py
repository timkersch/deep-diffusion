import utils
from sklearn.neighbors import KNeighborsRegressor
import numpy as np
from matplotlib.pyplot import hist
import matplotlib.pyplot as plt


def knn(X_train, y_train, X_val, y_val, X_hpc, class_index=0):
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
	model = KNeighborsRegressor(n_neighbors=10, weights='distance', algorithm='brute', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=1)
	model.fit(X_train, y_train[:, class_index])

	# Print kNN scores for train and validation set to measure how good the fit is
	print('Score train: ' + str(model.score(X_train, y_train[:, class_index])))
	print('Score val: ' + str(model.score(X_val, y_val[:, class_index])))

	# Make prediction on the HPC set
	predictions = model.predict(X_hpc)
	print(predictions)
	print('Min: ' + str(np.min(predictions)))
	print('Mean: ' + str(np.mean(predictions)))
	print('Max: ' + str(np.max(predictions)))

	n, bins, patches = hist(predictions, bins='auto', range=None, normed=False, weights=None, cumulative=False, bottom=None)
	print(n)
	print(bins)

	plt.show()


if __name__ == '__main__':
	# Load the generated dataset
	X_train, y_train, X_val, y_val = utils.get_param_eval_data(split_ratio=0.7)

	# Load 10000 randomly sampeld HPC voxels
	X_hpc = utils.get_hpc_data(10000)

	# Run on cylinder radius, i.e class index 0
	print('Cylinder radius:')
	knn(X_train, y_train, X_val, y_val, X_hpc, 0)

	print('')

	# Run on cylinder separation, i.e class index 1
	print('Cylinder separation:')
	knn(X_train, y_train, X_val, y_val, X_hpc, 1)
