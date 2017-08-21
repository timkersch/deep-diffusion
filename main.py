import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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

sys.setrecursionlimit(50000)


def train(model_id, train_set, validation_set, config, super_dir='models/'):
	input_var = T.fmatrix('inputs')
	target_var = T.fmatrix('targets')

	model_id = str(model_id)
	if not os.path.exists(super_dir + model_id):
		try:
			os.makedirs(super_dir + model_id)
		except OSError as exc:
			if exc.errno != errno.EEXIST:
				raise
	dir = super_dir + model_id + '/'

	# Write the config file
	with open(dir + 'config.json', 'w') as outfile:
		json.dump(config, outfile, sort_keys=True, indent=4)

	# Open file for appending output
	outfile = open(dir + 'out.txt', 'a')
	print_and_append('Training network with {} training samples and {} validation samples'.format(train_set[0].shape[0], validation_set[0].shape[0]), outfile)

	# Create neural network model
	network = FCNet(input_var, target_var, config)
	network.train(train_set[0], train_set[1], validation_set[0], validation_set[1], outfile=outfile)

	train_pred = network.predict(train_set[0])
	validation_pred = network.predict(validation_set[0])

	val_mse = mse(validation_set[1], validation_pred)
	val_r2 = r2(validation_set[1], validation_pred)

	print_and_append('Training MSE: ' + str(mse(train_set[1], train_pred)), outfile)
	print_and_append('Validation MSE: ' + str(val_mse), outfile)
	print_and_append('Training R2: ' + str(r2(train_set[1], train_pred)), outfile)
	print_and_append('Validation R2: ' + str(val_r2), outfile)

	outfile.close()
	save(dir + 'model.p', network)

	# Make some plots
	utils.loss_plot(network.train_loss, network.val_loss, filename=dir + 'loss-plot', zoomed=False)

	indices = np.random.choice(validation_set[1].shape[0], 1000)
	utils.diff_plot(validation_set[1][indices], validation_pred[indices], filename=dir + 'validation-diff-plot')
	utils.diff_plot(train_set[1][indices], train_pred[indices], filename=dir + 'train-diff-plot')

	return network, val_mse, val_r2


def parameter_search(dir='models/search/'):
	with open('config.json') as data_file:
		config = json.load(data_file)
	train_set, validation_set, test_set = dataset.load_dataset(config['no_dwis'], split_ratio=(0.6, 0.2, 0.2))

	learning_rates = 10 ** np.random.uniform(-6, -3.5, 10)
	batch_size = [128]
	scale_inputs = [True, False]
	batch_norm = [False, True]
	with_std = [False, True]

	layers = [
		[
			{
				"type": "fc",
				"units": 512
			},
			{
				"type": "fc",
				"units": 512
			},
			{
				"type": "fc",
				"units": 512
			},
			{
				"type": "fc",
				"units": 512
			},
			{
				"type": "fc",
				"units": 512
			},
			{
				"type": "fc",
				"units": 512
			},
		],
	]

	id_model_list = []
	lowest_mse = 1000
	best_index = -1
	index = 1

	no_configs = len(learning_rates)*len(scale_inputs)*len(layers)*len(batch_size)*len(with_std)*len(batch_norm)
	print "Beginning grid search with {} configurations".format(no_configs)
	for scale_in in scale_inputs:
		for w_std in with_std:
			for bn in batch_norm:
				for bs in batch_size:
					for layer in layers:
						for learning_rate in learning_rates:
							print "Fitting model {} of {} with l-rate: {}".format(index, no_configs, learning_rate)
							config['optimizer']['learning_rate'] = np.asscalar(learning_rate)
							config['hidden_layers'] = layer
							config['scale_inputs'] = scale_in
							config['batch_size'] = bs
							config['batch_norm'] = bn
							config['normalize']['with_std'] = w_std

							model, val_mse, val_r2 = train(super_dir=dir, train_set=train_set, validation_set=validation_set, model_id=index, config=config)

							id_model_list.append({'id': index, 'mse': np.asscalar(val_mse), 'r2': np.asscalar(val_r2)})

							if val_mse < lowest_mse:
								lowest_mse = val_mse
								best_index = index

							print 'Current best model is: {} with validation MSE: {} \n'.format(best_index, lowest_mse)

							index += 1

	axes = plt.gca()
	axes.set_ylim(0, 10 * np.median([k['mse'] for i, k in enumerate(id_model_list)]))
	plt.plot([k['id'] for i, k in enumerate(id_model_list)], [k['mse'] for i, k in enumerate(id_model_list)], 'bo')
	plt.ylabel('Validation MSE')
	plt.xlabel('Model ID')
	plt.savefig(dir + 'model-mse-plot')
	plt.close()

	id_model_list = sorted(id_model_list, key=lambda obj: obj['mse'])
	with open(dir + 'res.json', 'w') as outfile:
		json.dump(id_model_list, outfile, indent=4)
	print "Done... Best was model with index {} and validation MSE {}".format(best_index, lowest_mse)


def load(path):
	network = pickle.load(open(path, "rb"))
	return network


def save(path, network):
	pickle.dump(network, open(path, 'wb'))


def run_train():
	with open('config.json') as data_file:
		config = json.load(data_file)
	train_set, validation_set, test_set = dataset.load_dataset(config['no_dwis'], split_ratio=(0.6, 0.2, 0.2))
	model, _, _ = train(model_id='test', train_set=train_set, validation_set=validation_set, config=config)


if __name__ == '__main__':
	parameter_search('models/21-aug-1/‘)
	#run_train()
