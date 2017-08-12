import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import dataset
from networks.fc_network import FCNet
import theano.tensor as T
import json
import os
import errno
from utils import rmsd, print_and_append
import cPickle as pickle


def train(model_id, train_set, validation_set, config, super_dir='models/', show_plot=False):
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
	print_and_append('Training network with {} training samples and {} validation samples'.format(train_set[0].shape[0], validation_set[0].shape[0]), outfile)

	# Create neural network model
	network = FCNet(input_var, target_var, config)
	network.train(train_set[0], train_set[1], validation_set[0], validation_set[1], outfile=outfile)

	train_pred = network.predict(train_set[0])
	validation_pred = network.predict(validation_set[0])

	print_and_append('Training RMSE: ' + str(rmsd(train_pred, train_set[1])), outfile)
	print_and_append('Validation RMSE: ' + str(rmsd(validation_pred, validation_set[1])), outfile)

	save(dir + 'model.p', network)
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
	train_set, validation_set, test_set = dataset.load_dataset(config['no_dwis'], split_ratio=(0.8, 0.15, 0.05))

	learning_rates = [5e-6, 1e-5, 5e-5, 1e-4]
	batch_sizes = [128]
	early_stoppings = [0, 5, 10]
	scale_outputs = [False]

	lowest_rmsd = 1000
	best_index = -1

	id_model_list = []

	index = 1
	no_configs = len(learning_rates)*len(batch_sizes)*len(scale_outputs)*len(early_stoppings)
	print "Beginning grid search with {} configurations".format(no_configs)
	for scale_output in scale_outputs:
		for early_stopping in early_stoppings:
			for batch_size in batch_sizes:
				for learning_rate in learning_rates:
					print "Fitting model {} of {} with l-rate: {} batch-size: {} e-stopping: {} scale-out: {}".format(index, no_configs, learning_rate, batch_size, early_stopping, scale_output)
					config['optimizer']['learning_rate'] = learning_rate
					config['batch_size'] = batch_size
					config['early_stopping'] = early_stopping
					config['scale_outputs'] = scale_output

					model = train(super_dir=dir, train_set=train_set, validation_set=validation_set, model_id=index, config=config, show_plot=False)

					test_pred = model.predict(test_set[0])
					rms_distance = rmsd(test_pred, test_set[1])
					print 'Test RMSE: {} \n'.format(rms_distance)

					id_model_list.append({'id': index, 'rmse': rms_distance})

					if rms_distance < lowest_rmsd:
						lowest_rmsd = rms_distance
						best_index = index

					print 'Current best model is: {} with test RMSE: {} \n'.format(best_index, lowest_rmsd)

					index += 1

	id_model_list = sorted(id_model_list, key=lambda obj: obj['rmse'])
	with open(dir + 'res.json', 'w') as outfile:
		json.dump(id_model_list, outfile, indent=4)
	print "Done... Best was model with index {} and test RMSE {}".format(best_index, lowest_rmsd)


def load(path):
	network = pickle.load(open(path, "rb" ))
	return network


def save(path, network):
	pickle.dump(network, open(path, 'wb'))


def run_train():
	with open('config.json') as data_file:
		config = json.load(data_file)
	train_set, validation_set, test_set = dataset.load_dataset(config['no_dwis'], split_ratio=(0.8, 0.15, 0.05))
	train(model_id='22', train_set=train_set, validation_set=validation_set, config=config)

if __name__ == '__main__':
	parameter_search('models/search-relu2/')
