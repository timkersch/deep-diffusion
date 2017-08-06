import dataset
from classifiers.voxel_network import VoxNet
import theano.tensor as T
import json
import os
import errno
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from utils import rmsd, print_and_append


def train(model_id):
	# Prepare Theano variables for inputs and targets
	input_var = T.dmatrix('inputs')
	target_var = T.dmatrix('targets')

	with open('config.json') as data_file:
		config = json.load(data_file)

	train, validation, test = dataset.load_dataset(config['no_dwis'], split_ratio=(0.7, 0.2, 0.1))

	if config['scale_inputs']:
		in_scaler = MinMaxScaler()
		in_scaler.fit(train[0])
		train = in_scaler.transform(train[0]), train[1]
		validation = in_scaler.transform(validation[0]), validation[1]
		test = in_scaler.transform(test[0]), test[1]

	if config['scale_outputs']:
		out_scaler = MinMaxScaler()
		out_scaler.fit(train[1])
		train = train[0], out_scaler.transform(train[1])
		validation = validation[0], out_scaler.transform(validation[1])
		test = test[0], out_scaler.transform(test[1])

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
	network.train(train[0], train[1], validation[0], validation[1], no_epochs=config['no_epochs'], outfile=outfile)

	train_pred = network.predict(train[0])
	validation_pred = network.predict(validation[0])
	test_pred = network.predict(test[0])

	print_and_append('Training-set, Scaled RMSE: ' + str(rmsd(train_pred, train[1])), outfile)
	print_and_append('Training-set, Original RMSE: ' + str(rmsd(out_scaler.inverse_transform(train_pred), out_scaler.inverse_transform(train[1]))), outfile)

	print_and_append('Validation-set, Scaled RMSE: ' + str(rmsd(validation_pred, validation[1])), outfile)
	print_and_append('Validation-set, Original RMSE: ' + str(rmsd(out_scaler.inverse_transform(validation_pred), out_scaler.inverse_transform(validation[1]))), outfile)

	print_and_append('Test-set, Scaled RMSE: ' + str(rmsd(test_pred, test[1])), outfile)
	print_and_append('Test-set, Original RMSE: ' + str(rmsd(out_scaler.inverse_transform(test_pred), out_scaler.inverse_transform(test[1]))), outfile)

	print "True:"
	print(out_scaler.inverse_transform(test[1][0:5]))
	print "Pred:"
	print(out_scaler.inverse_transform(test_pred[0:5]))

	network.save(dir + 'model.npz')
	outfile.close()

	# Make some plots of loss and accuracy
	plt.plot(network.train_loss)
	plt.plot(network.val_loss)
	plt.ylabel('Loss')
	plt.xlabel('Epochs')
	plt.legend(['Train', 'Val'], loc='upper right')
	plt.show()
	plt.savefig(dir + 'loss-plot')
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
	train(model_id='21')


