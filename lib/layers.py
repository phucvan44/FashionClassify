import numpy as np


class Input:


    def forward(self, inputs):
        self.output = inputs


class Dense:

    def __init__(self, units, name=None, activation=None, input_shape=None):
        self.input_shape = input_shape
        self.units = units
        self.name = name
        self.activation = activation


    def setup(self):
        self.output_shape = (self.input_shape[-2], self.units)
        self.weights = 0.01 * np.random.randn(self.input_shape[-1], self.units)
        self.biases = np.zeros((1, self.units))
        self.caches = {
            "weight_momentums": np.zeros_like(self.weights),
            "weight_cache": np.zeros_like(self.weights),
            "bias_momentums": np.zeros_like(self.biases),
            "bias_cache": np.zeros_like(self.biases)
        }


    def forward(self, inputs):
        self.inputs = inputs
        self.output = (inputs@self.weights) + self.biases

        if self.activation != None:
            self.activation.forward(self.output)
            self.output = self.activation.output


    def backward(self, dvalues):
        if self.activation != None:
            self.activation.backward(dvalues)
            dvalues = self.activation.dinputs
        self.dweights = self.inputs.T@dvalues
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = dvalues@self.weights.T


    def get_parameters(self):
        return self.weights, self.biases


    def set_parameters(self, weights, biases):
        self.weights = weights
        self.biases = biases


class Conv2D:

    def __init__(self, filters, kernel_size=1, name=None, activation=None, input_shape=None):
        self.name = name
        self.input_shape = input_shape
        self.filters = filters
        self.kernel_size = kernel_size
        self.activation = activation
        self.weights = 0.01*np.random.randn(filters, kernel_size, kernel_size)


    def setup(self):
        self.output_shape = (self.input_shape[0], self.filters, self.input_shape[1]-self.kernel_size+1, self.input_shape[2]-self.kernel_size+1)


    def forward(self, inputs):
        self.inputs = inputs
        n_rows = inputs.shape[1]-self.kernel_size+1
        n_cols = inputs.shape[2]-self.kernel_size+1
        self.output = np.zeros((inputs.shape[0], self.filters, n_rows, n_cols))

        for id_filter in range(self.filters):
            for id_row in range(n_rows):
                for id_col in range(n_cols):
                    sub_inputs = inputs[::, id_row:id_row+self.kernel_size, id_col:id_col+self.kernel_size]
                    self.output[::, id_filter, id_row, id_col] = np.sum(np.sum(sub_inputs*self.weights[id_filter], axis=2), axis=1)

        if self.activation != None:
            self.activation.forward(self.output)
            self.output = self.activation.output


    def backward(self, dvalues):
        self.dinputs = dvalues


class MaxPooling2D:

    def __init__(self, name=None, pool_size=1, input_shape=None):
        self.name = name
        self.pool_size = pool_size
        self.input_shape = input_shape
        self.activation = None


    def setup(self):
        self.output_shape = (self.input_shape[0], self.input_shape[1], self.input_shape[2]-self.pool_size+1, self.input_shape[3]-self.pool_size+1)


    def forward(self, inputs):
        self.inputs = inputs
        n_layers = inputs.shape[0]
        n_rows = inputs.shape[2]-self.pool_size+1
        n_cols = inputs.shape[3]-self.pool_size+1
        self.output = np.zeros((n_layers, inputs.shape[1], n_rows, n_cols))
        for id_layer in range(n_layers):
            for id_row in range(n_rows):
                for id_col in range(n_cols):
                    sub_inputs = self.inputs[id_layer, ::, id_row:id_row+self.pool_size, id_col:id_col+self.pool_size]
                    self.output[id_layer, ::, id_row, id_col] = np.max(np.max(sub_inputs, axis=2), axis=1)


class Flatten:

    def __init__(self, input_shape=None):
        self.input_shape = input_shape
        self.name = None
        self.activation = None


    def setup(self):
        input_shape = list(self.input_shape)
        shape = [i for i in list(input_shape)[:-2]]
        shape.append(input_shape[-1]*input_shape[-2])
        self.output_shape = tuple(shape)


    def forward(self, inputs):
        self.inputs = inputs
        shape = [i for i in list(inputs.shape)[:-2]] + [inputs.shape[-1]*inputs.shape[-2]]
        # self.output = inputs.reshape(inputs.shape[0], -1)
        self.output = inputs.reshape(shape)


    def backward(self, dvalues):
        self.dinputs = dvalues