import utils
from sklearn.neighbors import KNeighborsRegressor
import numpy as np
from matplotlib.pyplot import hist
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
import dataset


def knn(X_train, y_train, X_val, y_val, X_hpc, class_index=0):
	model = KNeighborsRegressor(n_neighbors=10, weights='distance', algorithm='brute', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=1)
	model.fit(X_train, y_train[:, class_index])

	print('Score train: ' + str(model.score(X_train, y_train[:, class_index])))
	print('Score val: ' + str(model.score(X_val, y_val[:, class_index])))

	predictions = model.predict(X_hpc)
	print(predictions)
	print('Min: ' + str(np.min(predictions)))
	print('Mean: ' + str(np.mean(predictions)))
	print('Max: ' + str(np.max(predictions)))

	n, bins, patches = hist(predictions, bins='auto', range=None, normed=False, weights=None, cumulative=False, bottom=None)
	print(n)
	print(bins)

	plt.show()


def gp():
	gp = GaussianProcessRegressor(alpha=1e-10, optimizer='fmin_l_bfgs_b', n_restarts_optimizer=0, normalize_y=False, copy_X_train=True, random_state=None)
	train, validation, test = dataset.load_dataset(288, split_ratio=(0.6, 0.2, 0.2))
	gp = gp.fit(train[0], train[1])
	print gp.score(train[0], train[1])
	print gp.score(validation[0], validation[1])
	print gp.predict(test[0])
	print ""
	print test[1]
	return gp


if __name__ == '__main__':
	# Load the dataset
	X_train, y_train, X_val, y_val = utils.get_data(split_ratio=0.7)
	# Load HPC data

	print('Cylinder radius:')
	knn(X_train, y_train, X_val, y_val, X_hpc, 0)
	print('')
	print('Cylinder separation:')
	knn(X_train, y_train, X_val, y_val, X_hpc, 1)