import numpy as np


class Input:


    def forward(self, inputs):
        self.output = inputs


class Dense:

    def __init__(self, shape, name=None, activation=None):
        self.weights = 0.01*np.random.randn(shape[0], shape[1])
        self.biases = np.zeros((1, shape[1]))
        self.name = name
        self.activation = activation
        self.caches = {
            "weight_momentums": np.zeros_like(self.weights),
            "weight_cache": np.zeros_like(self.weights),
            "bias_momentums": np.zeros_like(self.biases),
            "bias_cache": np.zeros_like(self.biases)
        }


    def forward(self, inputs):
        self.inputs = inputs
        self.output = (inputs@self.weights) + self.biases


    def backward(self, dvalues):
        self.dweights = self.inputs.T@dvalues
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = dvalues@self.weights.T


    def get_parameters(self):
        return self.weights, self.biases


    def set_parameters(self, weights, biases):
        self.weights = weights
        self.biases = biases