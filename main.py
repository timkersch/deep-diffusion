import dataset
from classifiers.voxel_network import VoxNet
import theano.tensor as T
import json
import os
import errno
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.preprocessing import MinMaxScaler


def train(model_id, scale=True):
	# Prepare Theano variables for inputs and targets
	input_var = T.dmatrix('inputs')
	target_var = T.dmatrix('targets')

	with open('config.json') as data_file:
		config = json.load(data_file)

	train, validation, test = dataset.load_dataset(config['no_dwis'], split_ratio=(0.8, 0.2, 0))

	if scale:
		in_scaler = MinMaxScaler()
		in_scaler.fit(train[0])

		#out_scaler = MinMaxScaler()
		#out_scaler.fit(train[1])

		train = in_scaler.transform(train[0]), train[1] #out_scaler.transform(train[1])
		validation = in_scaler.transform(validation[0]), validation[1] #out_scaler.transform(validation[1])
		if (test[0].shape[0] > 0):
			test = in_scaler.transform(test[0]), test[1] #out_scaler.transform(test[1])

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


def gp():
	gp = GaussianProcessRegressor(alpha=1e-10, optimizer='fmin_l_bfgs_b', n_restarts_optimizer=0, normalize_y=False, copy_X_train=True, random_state=None)
	train, validation, test = dataset.load_dataset(288, split_ratio=(0.8, 0.199, 0.001))
	gp = gp.fit(train[0], train[1])
	print gp.score(train[0], train[1])
	print gp.score(validation[0], validation[1])
	print gp.predict(test[0])
	print ""
	print test[1]
	return gp

if __name__ == '__main__':
	train(model_id='14')


