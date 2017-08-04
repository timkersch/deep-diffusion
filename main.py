import dataset
from classifiers.voxel_network import VoxNet
import theano.tensor as T
import json
import os
import errno


def train():
	# Prepare Theano variables for inputs and targets
	input_var = T.dmatrix('inputs')
	target_var = T.dvector('targets')

	train, validation, test = dataset.load_dataset(split_ratio=(0.5, 0.5))
	model_id = '1'

	with open('config.json') as data_file:
		config = json.load(data_file)

	# Create neural network model
	network = VoxNet(input_var, target_var, config)
	network.train(train[0], train[1][:, 0], validation[0], validation[1][:, 0], config['no_epochs'])

	if not os.path.exists('models/' + model_id):
		try:
			os.makedirs('models/' + model_id)
		except OSError as exc:
			if exc.errno != errno.EEXIST:
				raise
	# Write the config file
	with open('models/' + model_id + '/config.json', 'w') as outfile:
		json.dump(config, outfile, sort_keys=True, indent=4)

	network.save('models/' + model_id + '/model.npz')
	return network


def load(model_id):
	# Prepare Theano variables for inputs and targets
	input_var = T.dmatrix('inputs')
	target_var = T.dvector('targets')

	path = 'models/' + str(model_id) + '/'
	with open(path + 'config.json') as data_file:
		config = json.load(data_file)

	# Create neural network model
	network = VoxNet(input_var, target_var, config)
	network.load(path + 'model.npz')
	return network


if __name__ == '__main__':
	pass


