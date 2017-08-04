import dataset
from classifiers.voxel_network import VoxNet
import theano.tensor as T


def train():
	# Prepare Theano variables for inputs and targets
	input_var = T.dmatrix('inputs')
	target_var = T.dvector('targets')

	train, validation, test = dataset.load_dataset()

	# Create neural network model
	network = VoxNet(input_var, target_var, batch_size=50)
	network.train(train[0], train[1], validation[0], validation[1], no_epochs=100)
	return network

if __name__ == '__main__':
	pass


