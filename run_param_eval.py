import matplotlib
matplotlib.use('Agg')
from matplotlib.ticker import FormatStrFormatter
import utils
from sklearn.neighbors import KNeighborsRegressor
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def heat_plot(matrix, filename, show=False):
	"""
	Do a heat plot in image space
	@param matrix: the HPC image data
	@param filename: the filename in which save 
	@param show: if the plot is to be shown or not
	"""
	fig = plt.figure()
	ax = fig.add_subplot(111)
	cax = ax.matshow(matrix, cmap=cm.gray)
	fig.colorbar(cax)

	if show:
		plt.show()
	else:
		plt.savefig(filename)
		plt.close()


def predictions_plot(targets, predictions, show=False):
	"""
	Plot predictions in histogram
	@param targets: the training data targets
	@param predictions: the HPC predictions made by kNN 
	@param show: if the plot is to be shown or saved
	"""
	fig, ax = plt.subplots()
	ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
	unique_values, counts = np.unique(np.log10(targets), return_counts=True)
	n, bins, patches = ax.hist(np.log10(predictions), rwidth=0.8, bins=30, range=(np.log10(np.min(targets)), np.log10(np.max(targets))), facecolor='green', alpha=0.7, label='HPC Predictions')
	ax.scatter(unique_values, counts, edgecolors=(0, 0, 0), label='Simulated training targets')

	plt.xlabel('Log base 10 Cylinder radius')
	plt.ylabel('Instance count')
	plt.title('Histogram of Cylinder radiuses')
	plt.xticks(bins)
	plt.setp(ax.get_xticklabels(), rotation=90, horizontalalignment='right')
	plt.grid(True)
	ax.legend()

	if show:
		plt.show()
	else:
		plt.savefig('./plots/predictions-plot.png')
		plt.close()


def knn(input, targets, hpc):
	"""
	Method used for fitting kNN to training data
	Used for comparing HPC data with generated data
	@param input: The input training data
	@param targets: The target training data
	@param hpc: The HPC input data to make predictions on
	"""
	# Fit model to training data
	model = KNeighborsRegressor(n_neighbors=5, weights='distance', algorithm='auto', metric='euclidean', n_jobs=4)
	model.fit(input, targets)

	# Make prediction on the HPC set
	predictions = model.predict(hpc)
	return predictions


def fit_subset(X_train, y_train):
	"""
	Load randomly sampeld HPC data and make predictions on them
	Also plot the predictions in a histogram
	@param X_train: The input training data to fit kNN with
	@param y_train: The target training data to fit kNN with 
	"""
	# Load randomly sampled HPC voxels
	X_hpc = utils.get_hpc_data(sample_size=50000)
	X_hpc = utils.filter_zeros(X_hpc)

	predictions = knn(X_train, y_train, X_hpc)
	predictions_plot(y_train, predictions, show=True)


def fit_full(X_train, y_train):
	"""
	Load all HPC data and make predictions for all of them
	Also plot a heat plot of results
	@param X_train: The input training data to fit kNN with
	@param y_train: The target training data to fit kNN with 
	"""
	# Load full HPC dataset
	X_hpc = utils.load_nib_data('./data/hpc/data.nii.gz')

	hpc_dimensions = X_hpc.shape
	X_hpc = X_hpc.reshape(-1, 288)

	noNonzeros = np.count_nonzero(X_hpc, axis=1)
	mask = np.where(noNonzeros == 0)

	print('Beginning kNN')
	predictions = knn(X_train, y_train, X_hpc)
	predictions[mask[0]] = 0

	spatialPredictions = predictions.reshape(hpc_dimensions[0], hpc_dimensions[1], hpc_dimensions[2])

	for i in range(0, hpc_dimensions[2]):
		heat_plot(spatialPredictions[:, :, i], './plots/heat-plot-z-slice-' + str(i) + '.png', show=False)

	predictions_plot(y_train, predictions, show=False)

if __name__ == '__main__':
	# Load the generated dataset
	X_train, y_train, _, _ = utils.get_param_eval_data(split_ratio=1.0)

	fit_full(X_train, y_train)
	#fit_subset(X_train, y_train)