import theano
import theano.tensor as T
import lasagne
import time
import numpy as np


class Network:

	def __init__(self, input_var, target_var, batch_size=50):
		self.batch_size = batch_size

		l_in = lasagne.layers.InputLayer(shape=(batch_size, 288), input_var=input_var)
		l_in_drop = lasagne.layers.dropout(l_in, p=0.2)
		l_hid1 = lasagne.layers.DenseLayer(l_in_drop, num_units=150, nonlinearity=lasagne.nonlinearities.rectify)
		l_hid1_drop = lasagne.layers.dropout(l_hid1, p=0.5)
		l_hid2 = lasagne.layers.DenseLayer(l_hid1_drop, num_units=150, nonlinearity=lasagne.nonlinearities.rectify)
		l_hid2_drop = lasagne.layers.dropout(l_hid2, p=0.5)
		l_out = lasagne.layers.DenseLayer(l_hid2_drop, 2, nonlinearity=lasagne.nonlinearities.linear)

		self.network = l_out

		prediction = lasagne.layers.get_output(self.network)
		loss = lasagne.objectives.squared_error(prediction, target_var)
		loss = loss.mean()
		params = lasagne.layers.get_all_params(self.network, trainable=True)
		updates = lasagne.updates.adam(loss, params, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8)

		test_prediction = lasagne.layers.get_output(self.network, deterministic=True)
		test_loss = lasagne.objectives.squared_error(test_prediction, target_var)
		test_loss = test_loss.mean()
		test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var), dtype=theano.config.floatX)

		self.train_forward = theano.function([input_var, target_var], loss, updates=updates)
		self.val_forward = theano.function([input_var, target_var], [test_loss, test_acc])
		self.predict = theano.function([input_var], [test_prediction])

	def predict(self, data):
		return self.predict(data)

	def train(self, X_train, y_train, X_val, y_val, no_epochs=100, shuffle=True, log_nth=None):
		for epoch in xrange(no_epochs):
			start_time = time.time()
			print('Epoch {} of {}'.format(epoch + 1 , no_epochs))

			train_err, train_batches = self._train(X_train, y_train, shuffle=shuffle, log_nth=log_nth)
			print("Epoch {} of {} took {:.3f}s".format(epoch + 1, no_epochs, time.time() - start_time))
			print("  training loss:\t\t{:.6f}".format(train_err / train_batches))

			#val_acc, val_err, val_batches = self._val(X_val, y_val, shuffle=shuffle)
			#print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
			#print("  validation accuracy:\t\t{:.2f} %".format(val_acc / val_batches * 100))
			print("")

	def _train(self, X, y, shuffle=True, log_nth=None):
		train_err = 0
		batch_index = 0
		for batch in self._iterate_minibatches(X, y, shuffle=shuffle):
			inputs, targets = batch
			train_err += self.train_forward(inputs, targets)
			if log_nth is not None and batch_index % log_nth == 0:
				print('Iteration {}'.format(batch_index + 1))
				print("  training loss:\t\t{:.6f}".format(train_err / batch_index + 1))
			batch_index += 1
		return train_err, batch_index

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