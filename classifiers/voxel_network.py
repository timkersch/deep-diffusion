from __future__ import division
import theano
import theano.tensor as T
import lasagne
import time
import numpy as np


class VoxNet:

	def __init__(self, input_var, target_var, config):
		self.config = config
		self.batch_size = config['batch_size']

		self.val_loss = []
		self.train_loss = []
		self.val_acc = []
		self.train_acc = []

		l_in = lasagne.layers.InputLayer(shape=(self.batch_size, config['no_dwis']), input_var=input_var)

		hidden_layers = config['hidden_layers']
		prev_layer = l_in
		for layer in hidden_layers:
			if layer['type'] == 'dropout':
				prev_layer = lasagne.layers.DropoutLayer(prev_layer, p=layer['p'])
			elif layer['type'] == 'fc':
				layer = lasagne.layers.DenseLayer(prev_layer, num_units=layer['units'], W=lasagne.init.Normal(std=1E-8, mean=0.0), nonlinearity=lasagne.nonlinearities.sigmoid)
				if config['batch_norm'] == True:
					prev_layer = lasagne.layers.batch_norm(layer)
				else:
					prev_layer = layer

		l_out = lasagne.layers.DenseLayer(prev_layer, 1, nonlinearity=lasagne.nonlinearities.linear)
		self.network = l_out

		prediction = lasagne.layers.get_output(self.network)
		loss = lasagne.objectives.squared_error(prediction, target_var)
		loss = loss.mean()
		params = lasagne.layers.get_all_params(self.network, trainable=True)

		if config['optimizer']['type'] == 'adam':
			updates = lasagne.updates.adam(loss, params, config['optimizer']['learning_rate'], config['optimizer']['beta1'], config['optimizer']['beta2'], config['optimizer']['epsilon'])
		elif config['optimizer']['type'] == 'momentum':
			updates = lasagne.updates.momentum(loss, params, config['optimizer']['learning_rate'], config['optimizer']['momentum'])

		test_prediction = lasagne.layers.get_output(self.network, deterministic=True)
		test_loss = lasagne.objectives.squared_error(test_prediction, target_var)
		test_loss = test_loss.mean()

		u_train = T.sum(T.pow(T.sub(target_var, prediction.reshape(target_var.shape)), 2))
		u_test = T.sum(T.pow(T.sub(target_var, test_prediction.reshape(target_var.shape)), 2))
		v = T.sum(T.pow(T.sub(target_var, T.mean(target_var, axis=0)), 2))

		train_acc = 1 - (u_train/v)
		test_acc = 1 - (u_test/v)

		self.train_forward = theano.function([input_var, target_var], [loss, train_acc], updates=updates)
		self.val_forward = theano.function([input_var, target_var], [test_loss, test_acc])

		self.predict_fun = theano.function([input_var], test_prediction)

	def predict(self, data):
		return self.predict_fun(data)

	def reset(self):
		self.train_loss = []
		self.val_loss = []

	def train(self, X_train, y_train, X_val, y_val, outfile=None, no_epochs=100, shuffle=True, log_nth=None):
		for epoch in xrange(no_epochs):
			start_time = time.time()

			train_acc, train_err, train_batches = self._train(X_train, y_train, shuffle=shuffle, log_nth=log_nth)
			self.train_loss.append(np.log(train_err))
			self.train_acc.append(train_acc)
			self._print_and_append("Epoch {} of {} took {:.3f}s".format(epoch + 1, no_epochs, time.time() - start_time), outfile)
			self._print_and_append("  training loss:\t\t{:.6f}".format(train_err / train_batches), outfile)
			self._print_and_append("  training accuracy:\t\t{:.6f}".format(train_acc / train_batches), outfile)

			val_acc, val_err, val_batches = self._val(X_val, y_val, shuffle=shuffle)
			self.val_loss.append(np.log(val_err))
			self.val_acc.append(val_acc)
			self._print_and_append("  validation loss:\t\t{:.6f}".format(val_err / val_batches), outfile)
			self._print_and_append("  validation accuracy:\t\t{:.6f}".format(val_acc / val_batches), outfile, new_line=True)

	def _train(self, X, y, shuffle=True, log_nth=None):
		train_err = 0
		train_acc = 0
		batch_index = 0
		for batch in self._iterate_minibatches(X, y, shuffle=shuffle):
			inputs, targets = batch
			err, acc = self.train_forward(inputs, targets)
			train_err += err
			train_acc += acc
			if log_nth is not None and batch_index % log_nth == 0:
				print('Iteration {}'.format(batch_index + 1))
				print("  training loss:\t\t{:.6f}".format(train_err / batch_index + 1))
			batch_index += 1
		return train_acc, train_err, batch_index

	def _val(self, X, y, shuffle=True):
		val_err = 0
		val_acc = 0
		batch_index = 0
		for batch in self._iterate_minibatches(X, y, shuffle=shuffle):
			inputs, targets = batch
			err, acc = self.val_forward(inputs, targets)
			val_err += err
			val_acc += acc
			batch_index += 1
		return val_acc, val_err, batch_index

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

	def _print_and_append(self, string, outfile, new_line=False):
		if outfile is not None:
			outfile.write(string)
			outfile.write('\n')
			if new_line:
				outfile.write('\n')

		print(string)
		if new_line:
			print '\n'
