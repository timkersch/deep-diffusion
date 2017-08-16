import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import dataset
from networks.fc_network import FCNet
import theano.tensor as T
import json
import os
import errno
from utils import rmse, mae, r2, print_and_append
import cPickle as pickle
import sys
import numpy as np

sys.setrecursionlimit(50000)


def train(model_id, train_set, validation_set, config, super_dir='models/', show_plot=False):
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

	print_and_append('Training RMSE: ' + str(rmse(train_set[1], train_pred)), outfile)
	print_and_append('Validation RMSE: ' + str(rmse(validation_set[1], validation_pred)), outfile)

	save(dir + 'model.p', network)

	# Make some plots of loss and accuracy
	plt.plot(network.train_loss)
	plt.plot(network.val_loss)
	plt.ylabel('Loss')
	plt.xlabel('Epochs')
	plt.legend(['Train', 'Val'], loc='upper right')
	if show_plot:
		plt.show()
	plt.savefig(dir + 'loss-plot')
	plt.close()

	return network, outfile


def parameter_search(dir='models/search/'):
	with open('config.json') as data_file:
		config = json.load(data_file)
	train_set, validation_set, test_set = dataset.load_dataset(config['no_dwis'], split_ratio=(0.6, 0.2, 0.2))

	learning_rates = 10 ** np.random.uniform(-6, -3, 20)
	batch_norms = [False, True]
	with_stds = [False, True]
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
			}
		],

		[
			{
				"type": "fc",
				"units": 150
			},
			{
				"type": "fc",
				"units": 150
			},
			{
				"type": "fc",
				"units": 150
			}
		],

		[
			{
				"type": "fc",
				"units": 512
			},
			{
				"type": "fc",
				"units": 256
			},
			{
				"type": "fc",
				"units": 128
			},
			{
				"type": "fc",
				"units": 256
			},
			{
				"type": "fc",
				"units": 512
			}
		],

		[
			{
				"type": "fc",
				"units": 128
			},
			{
				"type": "fc",
				"units": 256
			},
			{
				"type": "fc",
				"units": 512
			},
			{
				"type": "fc",
				"units": 256
			},
			{
				"type": "fc",
				"units": 128
			}
		],
	]

	id_model_list = []
	lowest_rmse = 1000
	best_index = -1
	index = 1

	no_configs = len(learning_rates)*len(batch_norms)*len(with_stds)*len(layers)
	print "Beginning grid search with {} configurations".format(no_configs)
	for batch_norm in batch_norms:
		for with_std in with_stds:
			for layer in layers:
				for learning_rate in learning_rates:
					print "Fitting model {} of {} with l-rate: {}".format(index, no_configs, learning_rate)
					config['optimizer']['learning_rate'] = np.asscalar(learning_rate)
					config['normalize']['with_std'] = with_std
					config['batch_norm'] = batch_norm
					config['hidden_layers'] = layer

					model, outfile = train(super_dir=dir, train_set=train_set, validation_set=validation_set, model_id=index, config=config, show_plot=False)

					test_pred = model.predict(test_set[0])
					rms_error = rmse(test_set[1], test_pred)
					ma_error = mae(test_set[1], test_pred)
					r2_score = r2(test_set[1], test_pred)

					id_model_list.append({'id': index, 'rmse': np.asscalar(rms_error), 'mae': np.asscalar(ma_error), 'r2': np.asscalar(r2_score)})

					print_and_append('Test RMSE: {}'.format(rms_error), outfile)
					print_and_append('Test MAE: {}'.format(ma_error), outfile)
					print_and_append('Test R2: {} \n'.format(r2_score), outfile)
					outfile.close()

					if rms_error < lowest_rmse:
						lowest_rmse = rms_error
						best_index = index

					print 'Current best model is: {} with test RMSE: {} \n'.format(best_index, lowest_rmse)

					index += 1

	plt.plot([k['id'] for i, k in enumerate(id_model_list)], [k['rmse'] for i, k in enumerate(id_model_list)], 'bo')
	plt.ylabel('Test RMSE')
	plt.xlabel('Model ID')
	plt.savefig(dir + 'model-rmse-plot')
	plt.close()

	id_model_list = sorted(id_model_list, key=lambda obj: obj['rmse'])
	with open(dir + 'res.json', 'w') as outfile:
		json.dump(id_model_list, outfile, indent=4)
	print "Done... Best was model with index {} and test RMSE {}".format(best_index, lowest_rmse)


def load(path):
	network = pickle.load(open(path, "rb"))
	return network


def save(path, network):
	pickle.dump(network, open(path, 'wb'))


def run_train():
	with open('config.json') as data_file:
		config = json.load(data_file)
	train_set, validation_set, test_set = dataset.load_dataset(config['no_dwis'], split_ratio=(0.6, 0.2, 0.2))
	model, outfile = train(model_id='test', train_set=train_set, validation_set=validation_set, config=config)
	outfile.close()

if __name__ == '__main__':
	parameter_search('models/big-search/')
	#run_train()
