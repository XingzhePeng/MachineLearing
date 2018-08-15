# coding=utf-8

import gzip
import pickle
import numpy as np


class DataLoader(object):
    def __init__(self):
        with gzip.open('../data/mnist.pkl.gz', 'rb') as data_bin:
            self.training_data, self.valid_data, self.test_data =\
                pickle.load(data_bin, encoding='bytes')

    def get_formatted_data(self):
        training_data = list(zip(self.training_data[0],
                                 [self.reshape_gt(gt) for gt in self.training_data[1]]))
        valid_data = list(zip([x.reshape((784, 1)) for x in self.valid_data[0]],
                              self.valid_data[1]))
        test_data = list(zip([x.reshape((784, 1)) for x in self.test_data[0]],
                             self.test_data[1]))
        return training_data, valid_data, test_data

    def reshape_gt(self, gt):
        gt_array = np.zeros(10)
        gt_array[gt] = 1.0
        return gt_array


class Network(object):
    '''
    Some notes:
    1: Difference may be there between random.shuffle and numpy.random.shuffle.
    2: The use of 'learning_rate/len(mini_batch)*w_d' or '(learning_rate/len(mini_batch))*w_d'
       may impact the performance of the network, why?
    3: Take care of the use of float or int, e.g., using 1.0 or 1.
    '''
    def __init__(self, layers):
        self.num_layers = len(layers)
        self.weights = [np.random.random((y, x)) for (x, y) in zip(layers[:-1], layers[1:])]
        self.bias = [np.random.random((x, 1)) for x in layers[1:]]

    def SGD(self, training_data, epochs, mini_batch_size, learning_rate, test_data=None):
        for epoch in range(epochs):
            np.random.shuffle(training_data)
            mini_batchs = [training_data[i:i+mini_batch_size]
                           for i in range(0, len(training_data), mini_batch_size)]
            for mini_batch in mini_batchs:
                self.update_mini_batch(mini_batch, learning_rate)
            if test_data:
                len_test_data = len(test_data)
                recognised_nums = sum([int(self.feed_forward(x) == y)
                                       for x, y in test_data])
                print('epoch {0}: {1} / {2} recognised'.format(epoch, recognised_nums, len_test_data))

    def update_mini_batch(self, mini_batch, learning_rate):
        x = np.array([x for (x, _) in mini_batch]).T
        y = np.array([y for (_, y) in mini_batch]).T

        delta_weights, delta_bias = self.backprop(x, y)
        self.weights = [w - (learning_rate/len(mini_batch))*w_d
                        for w, w_d in zip(self.weights, delta_weights)]
        self.bias = [b - (learning_rate/len(mini_batch))*b_d
                     for b, b_d in zip(self.bias, delta_bias)]

    def backprop(self, x, y):
        activation = x
        activations = [x]
        zs = []
        for w, b in zip(self.weights, self.bias):
            z = np.dot(w, activation) + b
            activation = self.activate(z)
            zs.append(z)
            activations.append(activation)
        delta_weights = [0] * len(self.weights)
        delta_bias = [0] * len(self.bias)

        error = (activations[-1] - y) * self.activate_derivative(zs[-1])
        delta_bias[-1] = error.sum(1).reshape((len(error), 1))
        delta_weights[-1] = error.dot(activations[-2].T)
        for i in range(2, self.num_layers):
            error = self.weights[-i + 1].T.dot(error) * self.activate_derivative(zs[-i])
            delta_bias[-i] = error.sum(1).reshape((len(error), 1))
            delta_weights[-i] = error.dot(activations[-i - 1].T)
        return delta_weights, delta_bias

    def feed_forward(self, input):
        for w, b in zip(self.weights, self.bias):
            input = self.activate(w.dot(input) + b)
        return input.argmax()

    def activate(self, input):
        return 1.0 / (1.0 + np.exp(-input))

    def activate_derivative(self, input):
        return self.activate(input) * (1 - self.activate(input))


class Trainer(object):
    def __init__(self, net_layers):
        self.dataloader = DataLoader()
        self.network = Network(net_layers)

    def train(self, epochs, mini_batch_size, learning_rate,):
        training_data, valid_data, test_data = self.dataloader.get_formatted_data()
        self.network.SGD(training_data, epochs, mini_batch_size, learning_rate, test_data)


if __name__ == '__main__':
    trainer = Trainer([784, 30, 10]) # layers
    trainer.train(30, 10, 3.0) # epochs, mini_batch_size, learning_rate
