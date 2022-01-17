import numpy as np
from functools import reduce
import sys


class Input:


    def forward(self, inputs):
        self.output = inputs


class Dense:

    def __init__(self, units, name=None, activation=None, input_shape=None, **kwargs):
        self.input_shape = input_shape
        self.units = units
        self.name = name
        self.activation = activation


    def setup(self, input_shape):
        if input_shape != None:
            self.input_shape = input_shape
        self.output_shape = (None, self.input_shape[-1], self.units)
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
        self.dweights = self.inputs.T @ dvalues
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = dvalues @ self.weights.T


    def get_parameters(self):
        return self.weights, self.biases


    def set_parameters(self, weights, biases):
        self.weights = weights
        self.biases = biases


class Conv2D:

    def __init__(self, filters, kernel_size, padding=0, stride=1, input_shape=None, activation=None, name=None,
                 **kwargs):
        self.filters = filters
        self.kernel_size = kernel_size
        self.padding = padding
        self.input_shape = input_shape
        self.activation = activation
        self.stride = stride
        self.name = name


    def setup(self, input_shape=None):
        if input_shape != None:
            self.input_shape = input_shape

        self.weights = np.random.randn(self.filters, self.input_shape[-3], self.kernel_size, self.kernel_size)
        self.biases = np.random.randn(self.filters)

        self.caches = {
            "weight_momentums": np.zeros_like(self.weights),
            "weight_cache": np.zeros_like(self.weights),
            "bias_momentums": np.zeros_like(self.biases),
            "bias_cache": np.zeros_like(self.biases)
        }

        self.output_shape = (
        None, self.filters, self.input_shape[-2] - self.kernel_size + 1, self.input_shape[-1] - self.kernel_size + 1)


    def get_indices(self, X_shape, kernel_size, stride, pad):
        n_points, n_filters, n_rows, n_cols = X_shape
        output_rows = int((n_rows + 2 * pad - kernel_size) / stride) + 1
        output_cols = int((n_cols + 2 * pad - kernel_size) / stride) + 1
        level1 = np.repeat(np.arange(kernel_size), kernel_size)
        level1 = np.tile(level1, n_filters)
        all_levels = stride * np.repeat(np.arange(output_rows), output_cols)
        idx1 = level1.reshape(-1, 1) + all_levels.reshape(1, -1)
        slide1 = np.tile(np.arange(kernel_size), kernel_size)
        slide1 = np.tile(slide1, n_filters)
        all_slides = stride * np.tile(np.arange(output_cols), output_rows)
        idx2 = slide1.reshape(-1, 1) + all_slides.reshape(1, -1)
        idx3 = np.repeat(np.arange(n_filters), kernel_size * kernel_size).reshape(-1, 1)
        return idx1, idx2, idx3


    def forward_col(self, inputs, kernel_size, stride, pad):
        X_padded = np.pad(inputs, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')
        i, j, d = self.get_indices(inputs.shape, kernel_size, stride, pad)
        cols = X_padded[:, d, i, j]
        cols = np.concatenate(cols, axis=-1)
        return cols


    def backward_col(self, inputs, X_shape, filter_size, stride, pad):
        N, D, H, W = X_shape
        H_padded, W_padded = H + 2 * pad, W + 2 * pad
        X_padded = np.zeros((N, D, H_padded, W_padded))

        i, j, d = self.get_indices(X_shape, filter_size, stride, pad)
        dX_col_reshaped = np.array(np.hsplit(inputs, N))
        np.add.at(X_padded, (slice(None), d, i, j), dX_col_reshaped)
        if pad == 0:
            return X_padded
        elif type(pad) is int:
            return X_padded[pad:-pad, pad:-pad, :, :]


    def forward(self, inputs):
        if len(inputs.shape) < 4:
            inputs = inputs.reshape(len(inputs), 1, *list(inputs.shape)[1:])
        self.inputs = inputs
        n_points, n_filters, n_rows, n_cols = inputs.shape

        rows = int((n_rows + 2 * self.padding - self.kernel_size) / self.stride) + 1
        cols = int((n_cols + 2 * self.padding - self.kernel_size) / self.stride) + 1

        X_col = self.forward_col(inputs, self.kernel_size, self.stride, self.padding)
        weights_col = self.weights.reshape((self.filters, -1))
        biases_col = self.biases.reshape(-1, 1)

        output = weights_col @ X_col + biases_col
        output = np.array(np.hsplit(output, n_points)).reshape((n_points, self.filters, rows, cols))
        self.caches["sets"] = [inputs, X_col, weights_col]
        self.output = output

        if self.activation != None:
            self.activation.forward(self.output)
            self.output = self.activation.output


    def backward(self, dvalues):
        if self.activation != None:
            self.activation.backward(dvalues)
            dvalues = self.activation.dinputs
        self.dinputs = dvalues
        X, X_col, w_col = self.caches["sets"]
        n_points, _, _, _ = X.shape

        self.dbiases = np.sum(dvalues, axis=(0, 2, 3))

        dout = dvalues.reshape(dvalues.shape[0]*dvalues.shape[1], dvalues.shape[2]*dvalues.shape[3])
        dout = np.array(np.vsplit(dout, n_points))
        dout = np.concatenate(dout, axis=-1)

        dX_col = w_col.T@dout
        dw_col = dout@X_col.T
        dX = self.backward_col(dX_col, X.shape, self.kernel_size, self.stride, self.padding)
        self.dweights = dw_col.reshape((dw_col.shape[0], self.input_shape[-3], self.kernel_size, self.kernel_size))


class MaxPooling2D:

    def __init__(self, pool_size, stride=None, padding=0, name=None, input_shape=None, **kwargs):
        self.pool_size = pool_size
        self.stride = stride
        self.padding = padding
        self.cache = None
        self.input_shape = input_shape
        self.name = name
        self.activation = None
        if self.stride == None:
            self.stride = self.pool_size


    def setup(self, input_shape):
        if input_shape != None:
            self.input_shape = input_shape
        shapeX = int((input_shape[-2] - self.pool_size) / self.stride) + 1
        shapeY = int((input_shape[-1] - self.pool_size) / self.stride) + 1
        self.output_shape = (None, *list(input_shape)[1:-2], shapeX, shapeY)


    def get_indices(self, X_shape, pool_size, stride, pad):
        n_points, n_filters, n_rows, n_cols = X_shape
        output_rows = int((n_rows + 2 * pad - pool_size) / stride) + 1
        output_cols = int((n_cols + 2 * pad - pool_size) / stride) + 1
        level1 = np.repeat(np.arange(pool_size), pool_size)
        level1 = np.tile(level1, n_filters)
        all_levels = stride * np.repeat(np.arange(output_rows), output_cols)
        idx1 = level1.reshape(-1, 1) + all_levels.reshape(1, -1)
        slide1 = np.tile(np.arange(pool_size), pool_size)
        slide1 = np.tile(slide1, n_filters)
        all_slides = stride * np.tile(np.arange(output_cols), output_rows)
        idx2 = slide1.reshape(-1, 1) + all_slides.reshape(1, -1)
        idx3 = np.repeat(np.arange(n_filters), pool_size * pool_size).reshape(-1, 1)
        return idx1, idx2, idx3


    def forward_col(self, inputs, pool_size, stride, pad):
        X_padded = np.pad(inputs, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')
        i, j, d = self.get_indices(inputs.shape, pool_size, stride, pad)
        cols = X_padded[:, d, i, j]
        cols = np.concatenate(cols, axis=-1)
        return cols


    def backward_col(self, inputs, X_shape, pool_size, stride, pad):
        N, D, H, W = X_shape
        H_padded, W_padded = H + 2 * pad, W + 2 * pad
        X_padded = np.zeros((N, D, H_padded, W_padded))

        i, j, d = self.get_indices(X_shape, pool_size, stride, pad)
        dX_col_reshaped = np.array(np.hsplit(inputs, N))
        np.add.at(X_padded, (slice(None), d, i, j), dX_col_reshaped)
        if pad == 0:
            return X_padded
        elif type(pad) is int:
            return X_padded[pad:-pad, pad:-pad, :, :]


    def forward(self, inputs):
        if len(inputs.shape) < 4:
            inputs = inputs.reshape(len(inputs), 1, *list(inputs.shape)[1:])
        self.inputs = inputs

        self.cache = inputs
        n_points, n_filters, n_rows, n_cols = inputs.shape

        rows = int((n_rows + 2 * self.padding - self.pool_size) / self.stride) + 1
        cols = int((n_cols + 2 * self.padding - self.pool_size) / self.stride) + 1

        X_col = self.forward_col(inputs, self.pool_size, self.stride, self.padding)
        X_col = X_col.reshape(n_filters, X_col.shape[0] // n_filters, -1)
        output = np.max(X_col, axis=1)
        output = np.array(np.hsplit(output, n_points))
        output = output.reshape(n_points, n_filters, rows, cols)
        self.output = output


    def backward(self, dvalues):
        self.dinputs = dvalues

        X = self.cache
        n_points, n_filters, n_rows, n_cols = X.shape
        rows = int((n_rows + 2 * self.padding - self.pool_size) / self.stride) + 1
        cols = int((n_cols + 2 * self.padding - self.pool_size) / self.stride) + 1

        dout_flatten = dvalues.reshape(n_filters, -1) / (self.pool_size * self.pool_size)
        dX_col = np.repeat(dout_flatten, self.pool_size * self.pool_size, axis=0)
        dX = self.backward_col(dX_col, X.shape, self.pool_size, self.stride, self.padding)
        dX = dX.reshape(n_points, -1)
        dX = np.array(np.hsplit(dX, n_filters))
        dX = dX.reshape(n_points, n_filters, n_rows, n_cols)
        self.dinputs = dX


class Flatten:

    def __init__(self, input_shape=None):
        self.input_shape = input_shape
        self.name = None
        self.activation = None

    def setup(self, input_shape):
        if input_shape != None:
            self.input_shape = input_shape
        input_shape = list(self.input_shape)
        self.output_shape = (None, reduce(lambda a, b: a*b, input_shape[1:], 1))


    def forward(self, inputs):
        self.inputs = inputs
        self.output = inputs.reshape(len(self.inputs), -1)


    def backward(self, dvalues):
        self.dinputs = dvalues