import theano.tensor as T
import utils
from classifiers.parameter_network import Network

# Load the dataset
X_train, y_train, X_val, y_val = utils.get_data(split_ratio=0.7)

# Prepare Theano variables for inputs and targets
input_var = T.dmatrix('inputs')
target_var = T.dvector('targets')


def regress_rad():
	# Create neural network model
	network = Network(input_var, target_var, batch_size=50)
	network.train(X_train, y_train[:, 0], X_val, y_val[:, 0], no_epochs=100)
	return network


def regress_sep():
	# Create neural network model
	network = Network(input_var, target_var, batch_size=50)
	network.train(X_train, y_train[:, 1], X_val, y_val[:, 1], no_epochs=100)
	return network

if __name__ == '__main__':
	regress_rad()