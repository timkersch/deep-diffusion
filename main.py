import matplotlib
matplotlib.use('Agg')
import dataset
from networks.fc_network import FCNet
import theano.tensor as T
import json
import os
import errno
from utils import mse, r2, print_and_append
import cPickle as pickle
import sys
import utils
import numpy as np
import argparse
from generate_data import run
import datetime

sys.setrecursionlimit(50000)


def train(train_set, validation_set, config='./config.json', model_path='models/model/'):
	"""
	Creates and calls a network to perform training	
	@param train_set: tuple holding the training set (X, y)
	@param validation_set: tuple holding the validation set (X, y)
	@param config: configuration file holding network settings
	@param model_path: Path where trained model should be saved
	@return: tuple - (network, validation_mse, validation_r2)
	"""

	T_input_var = T.fmatrix('inputs')
	T_target_var = T.fmatrix('targets')

	if not os.path.exists(model_path):
		try:
			os.makedirs(model_path)
		except OSError as exc:
			if exc.errno != errno.EEXIST:
				raise

	if not model_path.endswith('/'):
		model_path += '/'

	# Write the config file
	with open(model_path + 'config.json', 'w') as outfile:
		json.dump(config, outfile, sort_keys=True, indent=4)

	# Open file for appending output
	outfile = open(model_path + 'out.txt', 'a')
	print_and_append('Training network with {} training samples and {} validation samples'.format(train_set[0].shape[0], validation_set[0].shape[0]), outfile)

	# Create neural network model
	network = FCNet(T_input_var, T_target_var, config)
	network.train(train_set[0], train_set[1], validation_set[0], validation_set[1], outfile=outfile)

	# Make predictions on the full train and validation set
	train_pred = network.predict(train_set[0])
	validation_pred = network.predict(validation_set[0])

	# Compute MSE and R2 score for the validationset
	val_mse = mse(validation_set[1], validation_pred)
	val_r2 = r2(validation_set[1], validation_pred)

	# Append output
	print_and_append('Training MSE: ' + str(mse(train_set[1], train_pred)), outfile)
	print_and_append('Validation MSE: ' + str(val_mse), outfile)
	print_and_append('Training R2: ' + str(r2(train_set[1], train_pred)), outfile)
	print_and_append('Validation R2: ' + str(val_r2), outfile)

	# Close file and save model
	outfile.close()
	save(model_path + 'model.p', network)

	# Create loss plot
	utils.loss_plot(network.train_loss, network.val_loss, filename=model_path + 'loss-plot')

	# Create scatter plots for 1000 randomly sampled indicies from train and validation sets
	indices = np.random.choice(validation_set[1].shape[0], 1000)
	utils.diff_plot(validation_set[1][indices], validation_pred[indices], filename=model_path + 'validation-diff-plot')
	utils.diff_plot(train_set[1][indices], train_pred[indices], filename=model_path + 'train-diff-plot')
	utils.residual_plot(validation_set[1][indices], validation_pred[indices], filename=model_path + 'validation-residual-plot')
	utils.residual_plot(train_set[1][indices], train_pred[indices], filename=model_path + 'train-residual-plot')

	return network, val_mse, val_r2


def parameter_search():
	"""
	Train several networks to find good hyperparameters 	
	@return: nothing 
	"""
	dir='models/' + str(datetime.datetime.now().isoformat()) + '/'
	with open('config.json') as data_file:
		config = json.load(data_file)
	train_set, validation_set, test_set = dataset.load_dataset(config['no_dwis'], split_ratio=(0.6, 0.2, 0.2))

	# Define the search space
	learning_rates = np.sort(10 ** np.random.uniform(-5, -3, 5))
	batch_size = [64, 128, 256, 512, 1024]
	batch_norm = [False, True]

	id_model_list = []
	lowest_mse = 1000
	best_index = -1
	index = 1

	# Grid search over configurations
	no_configs = len(learning_rates)*len(batch_size)*len(batch_norm)
	print "Beginning grid search with {} configurations".format(no_configs)
	for bn in batch_norm:
		heat_matrix = np.empty(([len(learning_rates), len(batch_size)]))
		for i, lr in enumerate(learning_rates):
			for j, bs in enumerate(batch_size):
				print "Fitting model {} of {}".format(index, no_configs)

				config['learning_rate'] = lr
				config['batch_size'] = bs

				model, val_mse, val_r2 = train(train_set=train_set, validation_set=validation_set, model_path=dir + str(index), config=config)
				id_model_list.append({'id': index, 'mse': np.asscalar(val_mse), 'r2': np.asscalar(val_r2)})

				heat_matrix[i][j] = np.asscalar(val_r2)

				if val_mse < lowest_mse:
					lowest_mse = val_mse
					best_index = index

				print 'Current best model is: {} with validation MSE: {} \n'.format(best_index, lowest_mse)

				index += 1
		# Generate heat-plot
		utils.heat_plot(heat_matrix, dir + 'heat-plot-lr-vs-bs-bn:' + str(bn), learning_rates, batch_size, xLabel='Learning rate', yLabel='Batch size')

	# Generate scatter plot of each model vs mse
	utils.model_comp_plot(id_model_list, dir + 'model-mse-plot')

	# Save list of models, sorted by best performing
	id_model_list = sorted(id_model_list, key=lambda obj: obj['mse'])
	with open(dir + 'res.json', 'w') as outfile:
		json.dump(id_model_list, outfile, indent=4)
	print "Done... Best was model with index {} and validation MSE {}".format(best_index, lowest_mse)


def load(path):
	"""
	(OBSERVE) models trained on GPU can only be loaded on GPU
	Load network model from disk
	@param path: the path to the model
	@return: a FCNet object
	"""
	network = pickle.load(open(path, "rb"))
	return network


def save(path, network):
	"""
	Save neural network model to disk
	@param path: Where to save the network
	@param network: The network to save
	@return: nothing
	"""
	pickle.dump(network, open(path, 'wb'))


def run_train(config_path='./config.json', model_path='models/model/'):
	"""
	Helper method to start training
	@param config_path: path to the config file to use
	@param model_path: path where to save the trained model
	@return: nothing
	"""
	with open(config_path) as data_file:
		config = json.load(data_file)
	train_set, validation_set, test_set = dataset.load_dataset(config['no_dwis'], split_ratio=(0.6, 0.2, 0.2))
	model, _, _ = train(model_path=model_path, train_set=train_set, validation_set=validation_set, config=config)


# Parsing the command line happens here (See README.md)
if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	subparsers = parser.add_subparsers(help='commands')

	# Training
	training_parser = subparsers.add_parser('training', help='Train model')
	training_parser.add_argument('-m', action="store", help='Set path to save model', dest='model_dest')
	training_parser.add_argument('-c', action="store", help='Config file', dest='config_file')
	training_parser.set_defaults(which='training')

	# Inference
	inference_parser = subparsers.add_parser('inference', help='Perform inference with trained model')
	inference_parser.add_argument('-d', action="store", help='Data to perform inference on', dest='data_file')
	inference_parser.add_argument('-m', action="store", help='Model file', dest='model_file')
	inference_parser.add_argument('-f', action="store", help='Save file', dest='save_file')
	inference_parser.set_defaults(which='inference')

	# Generation
	generate_parser = subparsers.add_parser('generate', help='Generate data')
	generate_parser.add_argument('-i', type=int, action="store", help='No iterations to run', dest='no_iter')
	generate_parser.add_argument('-v', type=int, action="store", help='No voxels in every iteration', dest='no_voxels')
	generate_parser.set_defaults(which='generate')

	# Search
	search_parser = subparsers.add_parser('search', help='Search parameter')
	search_parser.set_defaults(which='search')

	args = parser.parse_args()

	if args.which == 'training':
		config = args.config_file
		model = args.model_dest
		run_train(config_path=config, model_path=model)
	elif args.which == 'inference':
		network = load(args.model_file)
		data = utils.to_voxels(utils.read_float(args.data_file))
		preds = network.predict(data)
		preds.tofile(args.save_file)
	elif args.which == 'generate':
		run(no_iter=args.no_iter, no_voxels=args.no_voxels)
	elif args.which == 'search':
		parameter_search()
	else:
		print 'Illegal argument'
