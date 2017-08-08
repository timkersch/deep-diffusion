import dataset
from networks.voxel_network import VoxNet
import theano.tensor as T
import json
import os
import errno
import matplotlib.pyplot as plt
from utils import rmsd, print_and_append


def train(model_id, train_set, validation_set, config, super_dir='models/', show_plot=False):
	# Prepare Theano variables for inputs and targets
	input_var = T.dmatrix('inputs')
	target_var = T.dmatrix('targets')

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

	# Create neural network model
	network = VoxNet(input_var, target_var, config)
	network.train(train_set[0], train_set[1], validation_set[0], validation_set[1], no_epochs=config['no_epochs'], outfile=outfile)

	train_pred = network.predict(train_set[0])
	validation_pred = network.predict(validation_set[0])

	print_and_append('Training-set, RMSE: ' + str(rmsd(train_pred, train_set[1])), outfile)
	print_and_append('Validation-set, RMSE: ' + str(rmsd(validation_pred, validation_set[1])), outfile)

	network.save(dir + 'model.npz')
	outfile.close()

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

	return network


def parameter_search(dir='models/search/'):
	with open('config.json') as data_file:
		config = json.load(data_file)
	train_set, validation_set, test_set = dataset.load_dataset(config['no_dwis'], split_ratio=(0.7, 0.2, 0.1))

	learning_rates = [1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3]
	batch_sizes = [32, 54, 128, 256]
	batch_norms = [False, True]
	scale_outputs = [True, False]
	early_stoppings = [5, 10]

	lowest_rmsd = 1000
	best_index = -1

	index = 1
	no_configs = len(learning_rates)*len(batch_sizes)*len(batch_norms)*len(scale_outputs)*len(early_stoppings)
	print "Beginning grid search with {} configurations".format(no_configs)
	for batch_norm in batch_norms:
		for scale_output in scale_outputs:
			for early_stopping in early_stoppings:
				for batch_size in batch_sizes:
					for learning_rate in learning_rates:
						print "Fitting model {} of {} with l-rate: {} batch-size: {} e-stopping: {} scale-out: {} batch-norm: {}".format(index, no_configs, learning_rate, batch_size, early_stopping, scale_output, batch_norm)
						config['optimizer']['learning_rate'] = learning_rate
						config['batch_size'] = batch_size
						config['early_stopping'] = early_stopping
						config['scale_outputs'] = scale_output
						config['batch_norm'] = batch_norm

						model = train(super_dir=dir, train_set=train_set, validation_set=validation_set, model_id=index, config=config, show_plot=False)

						test_pred = model.predict(test_set[0])
						rms_distance = rmsd(test_pred, test_set[1])
						print 'Test RMSE: {} \n'.format(rms_distance)

						if rms_distance < lowest_rmsd:
							lowest_rmsd = rms_distance
							best_index = index

						print 'Current best model is: {} with test RMSE: {} \n'.format(best_index, lowest_rmsd)

						index += 1

	print "Done... Best was model with index {} and test RMSE {}".format(best_index, lowest_rmsd)


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


def run_train():
	with open('config.json') as data_file:
		config = json.load(data_file)
	train_set, validation_set, test_set = dataset.load_dataset(config['no_dwis'], split_ratio=(0.7, 0.2, 0.1))
	train(model_id='21', train_set=train_set, validation_set=validation_set, config=config)

if __name__ == '__main__':
	parameter_search('models/search/')