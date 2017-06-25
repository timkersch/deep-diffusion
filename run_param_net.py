import theano.tensor as T
import utils
from classifiers.parameter_network import Network

# Load the dataset
X_train, y_train, X_val, y_val = utils.get_data(split_ratio=0.7)


def main():
	# Prepare Theano variables for inputs and targets
	input_var = T.matrix('inputs')
	target_var = T.matrix('targets')

	# Create neural network model
	network = Network(input_var, target_var, batch_size=50)
	network.train(X_train, y_train, X_val, y_val, no_epochs=200)
	return network