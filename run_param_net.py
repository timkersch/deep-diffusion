import theano.tensor as T
import utils
from classifiers.parameter_network import Network

# Load the dataset
X_train, y_train, X_val, y_val = utils.get_data(split_ratio=0.7)
# X_train = (X_train - X_train.mean()) / X_train.std()
# y_train = (y_train - y_train.mean()) / y_train.std()
# X_val = (X_val - X_val.mean()) / X_val.std()
# y_val = (y_val - y_val.mean()) / y_val.std()

# Prepare Theano variables for inputs and targets
input_var = T.dmatrix('inputs')
target_var = T.dvector('targets')


def regressRadius():
	# Create neural network model
	network = Network(input_var, target_var, batch_size=50)
	network.train(X_train, y_train[:, 0], X_val, y_val[:, 0], no_epochs=10)
	return network


def regressSeparation():
	# Create neural network model
	network = Network(input_var, target_var, batch_size=50)
	network.train(X_train, y_train[:, 1], X_val, y_val[:, 1], no_epochs=10)
	return network

if __name__ == '__main__':
	regressRadius()