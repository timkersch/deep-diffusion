import dataset
from classifiers.voxel_network import VoxNet
import theano.tensor as T
import json
import os
import errno
import matplotlib.pyplot as plt


def train(model_id):
	# Prepare Theano variables for inputs and targets
	input_var = T.dmatrix('inputs')
	target_var = T.dmatrix('targets')

	with open('config.json') as data_file:
		config = json.load(data_file)

	train, validation, test = dataset.load_dataset(config['no_dwis'], split_ratio=(0.8, 0.2, 0))

	if not os.path.exists('models/' + model_id):
		try:
			os.makedirs('models/' + model_id)
		except OSError as exc:
			if exc.errno != errno.EEXIST:
				raise
	dir = 'models/' + model_id + '/'

	# Write the config file
	with open(dir + 'config.json', 'w') as outfile:
		json.dump(config, outfile, sort_keys=True, indent=4)

	# Open file for appending output
	outfile = open(dir + 'out.txt', 'a')

	# Create neural network model
	network = VoxNet(input_var, target_var, config)
	network.train(train[0], train[1][:, 0].reshape(-1, 1), validation[0], validation[1][:, 0].reshape(-1, 1), no_epochs=config['no_epochs'], outfile=outfile)

	network.save(dir + 'model.npz')
	outfile.close()

	# Make some plots of loss and accuracy
	plt.plot(network.train_loss)
	plt.plot(network.val_loss)
	plt.ylabel('Log-loss')
	plt.xlabel('Epochs')
	plt.legend(['Train', 'Val'], loc='upper right')
	plt.savefig(dir + 'loss-plot')
	plt.close()

	plt.plot(network.train_acc)
	plt.plot(network.val_acc)
	plt.ylabel('R2-Score')
	plt.xlabel('Epochs')
	plt.legend(['Train', 'Val'], loc='upper right')
	plt.savefig(dir + 'r2-plot')
	plt.close()

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
	train(model_id='2')


