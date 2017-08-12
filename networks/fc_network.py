from __future__ import division
import theano
import lasagne
import numpy as np
from utils import print_and_append
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class FCNet:

	def __init__(self, input_var, target_var, config):
		self.config = config
		self.batch_size = config['batch_size']

		self.in_scaler = None
		self.out_scaler = None

		self.normalizer = None

		self.val_loss = []
		self.train_loss = []

		l_in = lasagne.layers.InputLayer(shape=(self.batch_size, config['no_dwis']), input_var=input_var)

		hidden_layers = config['hidden_layers']
		prev_layer = l_in
		for layer in hidden_layers:
			if layer['type'] == 'dropout':
				prev_layer = lasagne.layers.DropoutLayer(prev_layer, p=layer['p'])
			elif layer['type'] == 'fc':
				if config['activation_function'] == 'relu':
					layer = lasagne.layers.DenseLayer(prev_layer, num_units=layer['units'], W=lasagne.init.HeNormal('relu'), nonlinearity=lasagne.nonlinearities.rectify)
				elif config['activation_function'] == 'sigmoid':
					layer = lasagne.layers.DenseLayer(prev_layer, num_units=layer['units'], W=lasagne.init.GlorotNormal(1.0), nonlinearity=lasagne.nonlinearities.sigmoid)
				elif config ['activation_function'] == 'tanh':
					layer = lasagne.layers.DenseLayer(prev_layer, num_units=layer['units'], W=lasagne.init.GlorotNormal(1.0), nonlinearity=lasagne.nonlinearities.tanh)

				if config['batch_norm'] == True:
					prev_layer = lasagne.layers.batch_norm(layer)
				else:
					prev_layer = layer

		l_out = lasagne.layers.DenseLayer(prev_layer, 1, nonlinearity=lasagne.nonlinearities.linear)
		self.network = l_out

		prediction = lasagne.layers.get_output(self.network)
		loss = lasagne.objectives.squared_error(prediction, target_var).mean()

		params = lasagne.layers.get_all_params(self.network, trainable=True)

		if config['optimizer']['type'] == 'adam':
			updates = lasagne.updates.adam(loss, params, config['optimizer']['learning_rate'], config['optimizer']['beta1'], config['optimizer']['beta2'], config['optimizer']['epsilon'])
		elif config['optimizer']['type'] == 'momentum':
			updates = lasagne.updates.momentum(loss, params, config['optimizer']['learning_rate'], config['optimizer']['momentum'])

		test_prediction = lasagne.layers.get_output(self.network, deterministic=True)
		test_loss = lasagne.objectives.squared_error(test_prediction, target_var).mean()

		self.train_forward = theano.function([input_var, target_var], loss, updates=updates)
		self.val_forward = theano.function([input_var, target_var], test_loss)

		self.predict_fun = theano.function([input_var], test_prediction)

	def predict(self, data):
		if self.normalizer is not None:
			data = self.normalizer.transform(data)
		if self.in_scaler is not None:
			data = self.in_scaler.transform(data)
		pred = self.predict_fun(data)
		if self.out_scaler is not None:
			pred = self.out_scaler.inverse_transform(pred)
		return pred

	def reset(self):
		self.train_loss = []
		self.val_loss = []

	def train(self, X_train, y_train, X_val, y_val, outfile=None, no_epochs=100, shuffle=True, log_nth=None):
		early_stopping = self.config['early_stopping']
		prev_net = lasagne.layers.get_all_param_values(self.network)

		if self.config['normalize']:
			self.normalizer = StandardScaler()
			self.normalizer.fit(X_train)
			X_train = self.normalizer.transform(X_train)
			X_val = self.normalizer.transform(X_val)

		if self.config['scale_inputs']:
			self.in_scaler = MinMaxScaler()
			self.in_scaler.fit(X_train)
			X_train = self.in_scaler.transform(X_train)
			X_val = self.in_scaler.transform(X_val)

		if self.config['scale_outputs']:
			self.out_scaler = MinMaxScaler()
			self.out_scaler.fit(y_train)
			y_train = self.out_scaler.transform(y_train)
			y_val = self.out_scaler.transform(y_val)

		for epoch in xrange(no_epochs):
			print_and_append("Epoch {} of {}".format(epoch + 1, no_epochs), outfile)

			train_loss, train_batches = self._train(X_train, y_train, shuffle=shuffle, log_nth=log_nth)
			self.train_loss.append(train_loss)
			print_and_append("  training loss:\t\t{:.6E}".format((train_loss / train_batches)), outfile)

			val_loss, val_batches = self._val(X_val, y_val, shuffle=shuffle)
			self.val_loss.append(val_loss)
			print_and_append("  validation loss:\t\t{:.6E}".format((val_loss / val_batches)), outfile, new_line=True)

			if early_stopping >= 1 and (epoch+1) % early_stopping == 0 and len(self.val_loss) >= early_stopping * 2:
				prev_val_loss = np.mean(self.val_loss[-early_stopping*2:-early_stopping])
				current_val_loss = np.mean(self.val_loss[-early_stopping:])
				if current_val_loss > prev_val_loss:
					lasagne.layers.set_all_param_values(self.network, prev_net)
					print_and_append("Early stopping, val-loss increased over the last {} epochs from {} to {}".format(early_stopping, prev_val_loss, current_val_loss), outfile)
					return

			prev_net = lasagne.layers.get_all_param_values(self.network)

	def _train(self, X, y, shuffle=True, log_nth=None):
		train_loss = 0
		batch_index = 0
		for batch in self._iterate_minibatches(X, y, shuffle=shuffle):
			inputs, targets = batch
			loss = self.train_forward(inputs, targets)
			train_loss += loss
			if log_nth is not None and batch_index % log_nth == 0:
				print('Iteration {}'.format(batch_index + 1))
				print("  training loss:\t\t{:.20f}".format(train_loss / batch_index + 1))
			batch_index += 1

		return train_loss, batch_index

	def _val(self, X, y, shuffle=True):
		val_loss = 0
		batch_index = 0
		for batch in self._iterate_minibatches(X, y, shuffle=shuffle):
			inputs, targets = batch
			loss = self.val_forward(inputs, targets)
			val_loss += loss
			batch_index += 1
		return val_loss, batch_index

	def _iterate_minibatches(self, X, y, shuffle=True):
		assert self.batch_size <= X.shape[0]
		assert len(X) == len(y)
		if shuffle:
			indices = np.arange(len(X))
			np.random.shuffle(indices)
		for start_idx in range(0, len(X) - self.batch_size + 1, self.batch_size):
			if shuffle:
				excerpt = indices[start_idx:start_idx + self.batch_size]
			else:
				excerpt = slice(start_idx, start_idx + self.batch_size)
			yield X[excerpt], y[excerpt]

	def save(self, filename):
		np.savez(filename, *lasagne.layers.get_all_param_values(self.network))

	def load(self, filename):
		with np.load(filename) as f:
			param_values = [f['arr_%d' % i] for i in range(len(f.files))]
			lasagne.layers.set_all_param_values(self.network, param_values)
