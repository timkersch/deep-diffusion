import utils
from sklearn.neighbors import KNeighborsRegressor
import numpy as np
from matplotlib.pyplot import hist
import matplotlib.pyplot as plt


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


if __name__ == '__main__':
	# Load the dataset
	X_train, y_train, X_val, y_val = utils.get_data(split_ratio=0.7)
	# Load HPC data
	X_hpc = utils.get_pred_data(10000)

	print('Cylinder radius:')
	knn(X_train, y_train, X_val, y_val, X_hpc, 0)
	print('')
	print('Cylinder separation:')
	knn(X_train, y_train, X_val, y_val, X_hpc, 1)