from __future__ import division
import numpy as np
from matplotlib.pyplot import hist
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def get_param_eval_data(split_ratio=0.7):
	"""
	Method to load and split param_eval data 
	@param split_ratio: Ratio to split data into training and testing set 0.7 -> 70% training 30% testing 
	@return: (X_train, y_train, X_test, y_test)
	"""

	# Files used for simulated voxel files for searching
	#files = ['./data/search/1000voxels_uniform_p=0_rad=0.1E-6_sep=1.1E-6_HPC-scheme.bfloat',
	#		 './data/search/1000voxels_uniform_p=0_rad=0.1E-6_sep=2.1E-6_HPC-scheme.bfloat',
	#		 './data/search/1000voxels_uniform_p=0_rad=0.5E-6_sep=1.1E-6_HPC-scheme.bfloat',
	#		 './data/search/1000voxels_uniform_p=0_rad=1.5E-6_sep=3.1E-6_HPC-scheme.bfloat',
	#		 './data/search/1000voxels_uniform_p=0_rad=1E-6_sep=3.1E-6_HPC-scheme.bfloat',
	#		 './data/search/1000voxels_uniform_p=0_rad=2E-6_sep=4.1E-6_HPC-scheme.bfloat',
	#		 './data/search/1000voxels_uniform_p=0_rad=1.5E-7_sep=1.1E-6_HPC-scheme.bfloat',
	#		 './data/search/1000voxels_uniform_p=0_rad=1E-7_sep=1.1E-6_HPC-scheme.bfloat',
	#		 './data/search/1000voxels_uniform_p=0_rad=1E-8_sep=1.1E-6_HPC-scheme.bfloat',
	#		 './data/search/1000voxels_uniform_p=0_rad=2E-7_sep=1.1E-6_HPC-scheme.bfloat',
	#		 './data/search/1000voxels_uniform_p=0_rad=5E-8_sep=1.1E-6_HPC-scheme.bfloat']

	files = [
	'data/search/1000voxels_uniform_p=0_rad=0.5E-6_sep=1.1E-6_HPC-scheme.bfloat',
	'data/search/1000voxels_uniform_p=0_rad=1.5E-7_sep=1.1e-6_HPC-scheme.bfloat',
	'data/search/1000voxels_uniform_p=0_rad=1E-7_sep=1.1e-6_HPC-scheme.bfloat',
	'data/search/1000voxels_uniform_p=0_rad=1E-8_sep=1.1e-6_HPC-scheme.bfloat',
	'data/search/1000voxels_uniform_p=0_rad=1E-9_sep=1.1e-6_HPC-scheme.bfloat',
	'data/search/1000voxels_uniform_p=0_rad=1E-10_sep=1.1e-6_HPC-scheme.bfloat',
	'data/search/1000voxels_uniform_p=0_rad=2E-7_sep=1.1e-6_HPC-scheme.bfloat',
	'data/search/1000voxels_uniform_p=0_rad=2E-8_sep=1.1e-6_HPC-scheme.bfloat',
	'data/search/1000voxels_uniform_p=0_rad=5E-8_sep=1.1e-6_HPC-scheme.bfloat']

	# A list of targets for the simulations (Cylinder rad, Cylinder sep)
	targets = [(0.5E-6, 0),
			   (1.5E-7, 0),
			   (1E-7, 0),
			   (1E-8, 0),
			   (1E-9, 0),
			   (1E-10, 0),
			   (2E-7, 0),
			   (2E-8, 0),
			   (5E-8, 0)]

	X, y = _load_data(files, targets)
	split = int(X.shape[0] * split_ratio)
	indices = np.random.permutation(X.shape[0])
	training_idx, test_idx = indices[:split], indices[split:]
	return X[training_idx, :], y[training_idx, :], X[test_idx, :], y[test_idx, :]


def _load_data(file_list, target_list):
	"""
	Helper method to load param-eval data from files
	@param file_list: list of data-files  
	@param target_list: list of target-files
	@return: (X, y) tuple of data where X are inputs and y are targets
	"""
	X = np.empty((len(file_list) * 1000, 288))
	y = np.empty((len(file_list) * 1000, 2))

	start = 0
	end = 1000
	for i in xrange(0, len(file_list)):
		file = file_list[i]
		target_tuple = target_list[i]
		vals = to_voxels(read_float(file), skip_ones=True)

		X[start:end] = vals
		y[start:end] = np.array(target_tuple)

		start = end
		end = end + 1000

	return X, y


def get_hpc_data(filename='./data/hpc/50000_scanned_voxels.Bfloat', sample_size=None):
	"""
	Helper method for loading HPC data from disk
	@param filename: a string specifying the path to the binary HPC data
	@param sample_size: number, only return a sample of specified size  
	@return: 
	"""
	arr = to_voxels(read_float(filename))
	if sample_size is not None:
		np.random.shuffle(arr)
		return arr[0:sample_size, :]
	return arr


def plot_features(inputs):
	"""
	Helper method for plotting each feature in a histogram 
	@param inputs: the inputs to plot
	@return: nothing but shows a plot
	"""
	for i in range(0, inputs.shape[1]):
		x = inputs[:, i]
		hist(x, bins='auto', range=None, normed=False, weights=None, cumulative=False, bottom=None)
		plt.show()


def plot_targets(targets):
	"""
	Helper method for plotting targets in histogram
	@param targets: the targets to plot
	@return: nothing but shows a plot
	"""
	hist(targets, bins='auto', range=None, normed=False, weights=None, cumulative=False, bottom=None)
	plt.show()


def read_float(filename):
	"""
	Helper method for reading binary float file 
	@param filename: the filename to read
	@return: an array of the floats in the binary file
	"""
	f = open(filename, "r")
	arr = np.fromfile(f, dtype='>f4')
	return arr


def to_voxels(arr, channels=288, skip_ones=True):
	"""
	Helper method to convert 1D-array to voxel arranged data	
	@param arr: the 1D array to convert
	@param channels: number of DWIs, i.e the number of channels / features in the data
	@param skip_ones: if the 1x1x1 (voxel dimensions) should be skipped in result
	@return: either a n x 1 x 1 x 1 x channels array or a n x channels array
	"""
	no_samples = int(arr.size / channels)
	if skip_ones:
		return np.reshape(arr, (no_samples, channels))
	return np.reshape(arr, (no_samples, 1, 1, 1, channels))


def diff_plot(targets, predictions, filename, remove_outliers=False):
	"""
	Method that creates ad saves a scatter plot of targets vs predictions
	@param targets: the targets array
	@param predictions: the predictions array
	@param filename: the filename of where to save the plot
	@param remove_outliers: if outliers should be removed from plotting (to get better scale)
	@return: nothing
	"""
	if remove_outliers:
		indices = np.where(np.logical_not(np.logical_or(np.abs(predictions) > 10 * np.abs(targets), np.abs(predictions) < np.abs(targets) / 10.0)))
		targets = targets[indices]
		predictions = predictions[indices]

	if targets.shape[0] != 0:
		fig, ax = plt.subplots()
		fig.suptitle(str(targets.shape[0]) + ' samples, R2: ' + str(r2(targets, predictions)), fontsize=12)
		axes = plt.gca()
		axes.set_ylim(np.min(predictions), np.max(predictions))
		axes.set_xlim(np.min(targets), np.max(targets))
		ax.scatter(targets, predictions, edgecolors=(0, 0, 0))
		ax.set_xlabel('Targets')
		ax.set_ylabel('Predictions')
		ax.plot([targets.min(), targets.max()], [targets.min(), targets.max()], 'k--', lw=4)
		plt.savefig(filename)
		plt.close()


def loss_plot(train_loss, val_loss, filename):
	"""
	Method that creates and saves a loss plot 
	@param train_loss: array or list of the training loss
	@param val_loss: array or list of the validation loss
	@param filename: the filename of where to save the plot
	@return: nothing
	"""
	plt.plot(train_loss)
	plt.plot(val_loss)
	plt.ylabel('Loss')
	plt.xlabel('Epochs')
	plt.legend(['Train', 'Val'], loc='upper right')
	plt.savefig(filename)
	plt.close()


def residual_plot(targets, predictions, filename):
	"""
	Mehtod that creates and saves a residual plot
	@param targets: the targets array
	@param predictions: the predictions array
	@param filename: the filename of where to save the plot
	@return: nothing
	"""
	fig, ax = plt.subplots()
	fig.suptitle(str(targets.shape[0]) + ' samples, Residual Plot', fontsize=12)
	residuals = targets - predictions
	axes = plt.gca()
	axes.set_ylim(np.min(residuals), np.max(residuals))
	axes.set_xlim(np.min(predictions), np.max(predictions))
	ax.scatter(predictions, residuals, edgecolors=(0, 0, 0))
	ax.set_xlabel('Predictions')
	ax.set_ylabel('Residuals')
	plt.savefig(filename)
	plt.close()


def heat_plot(matrix, filename, xTicks, yTicks, xLabel='X', yLabel='Y'):
	"""
	Method that creates and saves a heat plot between two hyperparameters	
	@param matrix: a matrix of r2 scores 
	@param filename: the filename and directory of where to save the plot
	@param xTicks: the ticks to use on x-axis
	@param yTicks: the ticks to use on y-axis
	@param xLabel: the label describing the data in the x-axis 
	@param yLabel: the label describing the data in the y-axis
	@return: nothing
	"""
	fig = plt.figure()
	ax = fig.add_subplot(111)
	cax = ax.matshow(matrix, vmin=0, vmax=1)
	fig.colorbar(cax)
	ticks = np.arange(0, matrix.shape[0], 1)
	ax.set_xticks(ticks)
	ax.set_yticks(ticks)
	ax.set_xticklabels(xTicks)
	ax.set_yticklabels(yTicks)
	ax.set_xlabel(xLabel)
	ax.set_ylabel(yLabel)
	plt.savefig(filename)
	plt.close()


def model_comp_plot(id_model_list, filename):
	"""
	Creates and saves a plot of model id vs MSE
	@param id_model_list: a list of dicts holding the mse for each model id
	@param filename: the filename and directory where the plot should be saved
	@return: nothing
	"""
	axes = plt.gca()
	axes.set_ylim(0, 10 * np.median([k['mse'] for i, k in enumerate(id_model_list)]))
	plt.plot([k['id'] for i, k in enumerate(id_model_list)], [k['mse'] for i, k in enumerate(id_model_list)], 'bo')
	plt.ylabel('Validation MSE')
	plt.xlabel('Model ID')
	plt.savefig(filename)
	plt.close()


def r2(t, y):
	"""
	Method that computes R2 score
	@param t: targets array
	@param y: predictions array
	@return: the r2 score between t and y
	"""
	return r2_score(t, y)


def mae(t, y):
	"""
	Method that computes the mean absolute error	
	@param t: targets array
	@param y: predictions array
	@return: the mae between t and y
	"""
	return mean_absolute_error(t, y)


# Method that computes mean squared error
def mse(t, y, rmse=False):
	"""
	Method that computes the mean squared error
	@param t: targets array
	@param y: predictions array
	@param rmse: boolean, if true return root MSE
	@return: the mean squared erro between t and y
	"""
	mse = mean_squared_error(t, y)
	if rmse:
		mse = mse ** 0.5
	return mse


def print_and_append(string, outfile, new_line=False):
	"""
	Helper method for both printing and appending to an output file	
	@param string: the string to append
	@param outfile: the file to append to
	@param new_line: if True, include a space after appending 
	@return: nothing
	"""
	if outfile is not None:
		outfile.write(string)
		outfile.write('\n')
		if new_line:
			outfile.write('\n')

	print(string)
	if new_line:
		print '\n'