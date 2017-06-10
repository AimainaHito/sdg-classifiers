#!/bin/python3
import os

import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x, derivative=False):
    if derivative:
        return x * (1 - x)

    return 1 / (1 + np.exp(-x))


def relu(x, derivative=False):
    if derivative:
        return x > 0

    return np.maximum(0, x)


def tanh(x, derivative=False):
    if derivative:
        return 1 - x**2

    return np.tanh(x)


class NeuralNetwork:

    def __init__(self, layers):
        """
        Creates a new neural network from the given layer information

        :param layers: Should be a list of tuples in the form: ((number of inputs, number of outputs), activation function)
        """
        self.layers = []
        for num_inputs_and_neurons, function in layers:
            #create random weights in range [-1,1]
            self.layers.append([2 * np.random.random(num_inputs_and_neurons) - 1, function])

    def propagate(self, inputs):
        """
        Propagates an input through the network

        :param input: A unit vector or matrix of inputs

        :return: The outputs of each layer. The last element of the list is the network's output.
        """
        layer_outputs = [inputs]

        for weights, function in self.layers:
            layer_outputs.append(function(layer_outputs[-1].dot(weights)))

        return layer_outputs

    def train(self, inputs, labels, num_iterations=1000, learning_rate=1, plot_errors=False):
        """
        Trains the network on input and label matrices

        :param inputs: A unit vector or matrix of inputs
        :param inputs: A unit vector or matrix of labels (or expected output values in the case of multiclass classification)
        :param num_iterations: The number of training iterations
        :param learning_rate: The learning weight used for layer weight adjustments
        :param plot_errors: If true plots the errors of the training process per iteration before exiting the function
        """
        if plot_errors:
            errors = []

        for _ in range(num_iterations):
            #forward propagate inputs
            outputs = self.propagate(inputs)

            if plot_errors:
                errors.append((labels - outputs[-1]).sum())

            adjustments = []

            #calculate output layer error and multiply by the gradient
            previous_delta = (labels - outputs[-1]) * self.layers[-1][1](outputs[-1], derivative=True)
            #dot product with previous layer output
            adjustments.append(outputs[-2].T.dot(previous_delta))

            #loop backwards through the layers and outputs
            for i in range(1, len(outputs) - 1)[::-1]:
                #distribute error by dot product of previous error and current layer weights
                previous_delta = (previous_delta.dot(self.layers[i][0].T)) * self.layers[i][1](outputs[i], derivative=True)
                adjustments.append(outputs[i - 1].T.dot(previous_delta))

            #apply adjustments
            for layer, adjustment in zip(self.layers, adjustments[::-1]):
                layer[0] += learning_rate * adjustment

        if plot_errors:
            plt.plot(errors)
            plt.show()


def read_examples(folder):
    """
    Reads ascii art examples from a folder

    :param folder: Retrieves all files from this folder. Uses default file system ordering.
    """
    examples = []
    for name in os.listdir(folder):
        with open(folder + "/" + name, "r", encoding="utf-8") as f:
            label = int(f.readline().strip())
            data = []
            for line in f:
                data.extend([int(cell == '#') for cell in line.strip()])

            examples.append((data, label))

    return examples


def read_sets(folder):
    """
    Read training and test sets from a folder

    :param folder: Should contain a train and a test folder
    """
    return read_examples(folder + "/train"), read_examples(folder + "/test")


if __name__ == '__main__':
    np.random.seed(1) #seeding the random generator to get reproducible results

    network = NeuralNetwork([((36, 18), sigmoid), ((18, 9), sigmoid), ((9, 1), sigmoid)]) #((inputs, outputs), activation function)
    data, test = read_sets("trainingsets/asciiart")

    inputs = np.array([i[0] for i in data])
    expected = np.array([i[1] for i in data]).reshape(-1, 1) #turn 1D labels into a unit vector

    network.train(inputs, expected, 1000, 1)

    #print resulting weights
    print("Trained-----")
    print(*["{} : {}".format(w, i) for w, i in zip(network.propagate(inputs)[-1], expected.reshape(-1))], sep="\n")

    print("Test-----")
    test_inputs = np.array([i[0] for i in test])
    print(*["{} : {}".format(w, i[1]) for w, i in zip(network.propagate(test_inputs)[-1], test)], sep="\n")
