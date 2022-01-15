import numpy as np
import pickle
import copy
from .layers import Input, Dense, Conv2D
from .activation import Softmax, ReLU

class Model:

    def __init__(self):
        self.layers = []
        self.trainable_layers = []
        self.input_layer = Input()


    def add(self, layer):
        if len(self.layers) == 0:
            layer.setup()
            layer.prev = self.input_layer
        else:
            self.layers[-1].next = layer
            layer.input_shape = self.layers[-1].output_shape
            layer.setup()
            layer.prev = self.layers[-1]
        if hasattr(layer, 'weights') and not isinstance(layer, Conv2D):
            self.trainable_layers.append(layer)
        self.layers.append(layer)


    def compile(self, loss, optimizer, accuracy):
        if loss != None:
            self.loss = loss
        if optimizer != None:
            self.optimizer = optimizer
        if accuracy != None:
            self.accuracy = accuracy

        self.layers[-1].next = self.loss
        self.output_layer_activation = self.layers[-1].activation
        self.loss.remember_trainable_layers(self.trainable_layers)


    def progress(self, index, total, accuracy, loss):
        percent = ("{0:.2f}").format(100 * ((index) / total))
        filledLength = 30 * index // total
        if filledLength == 30:
            bar = "="*30
        else:
            bar = '=' * filledLength + ">" + '-' * (30 - filledLength-1)
        index = " "*(len(str(total)) - len(str(index))) + str(index)
        print('\r%s/%s: |%s| %s%%   accuracy: %.3f   loss: %.3f' % (index, total, bar, percent, accuracy, loss), end="\r")
        if int(index) == total:
            print()


    def fit(self, X, y, epochs=1, batch_size=32):
        train_steps = len(X) // batch_size
        if train_steps * batch_size < len(X):
            train_steps += 1

        for epoch in range(1, epochs+1):

            print(f'Epoch: {epoch}/{epochs}')

            self.loss.new_pass()
            self.accuracy.new_pass()

            for step in range(train_steps):

                batch_X = X[step*batch_size:(step+1)*batch_size]
                batch_y = y[step*batch_size:(step+1)*batch_size]

                output = self.forward(batch_X)
                loss = self.loss.calculate(output, batch_y)

                predictions = self.output_layer_activation.predictions(output)

                accuracy = self.accuracy.calculate(predictions,batch_y)

                self.backward(output, batch_y)

                self.optimizer.pre_update_params()
                for layer in self.trainable_layers:
                    self.optimizer.update_params(layer)
                self.optimizer.post_update_params()

                self.progress(step+1, train_steps, accuracy, loss)

            epoch_loss = self.loss.calculate_accumulated()
            epoch_accuracy = self.accuracy.calculate_accumulated()


    def predict(self, X, batch_size=None):
        prediction_steps = 1

        if batch_size is not None:
            prediction_steps = len(X) // batch_size
            if prediction_steps * batch_size < len(X):
                prediction_steps += 1
        output = []
        for step in range(prediction_steps):
            if batch_size is None:
                batch_X = X
            else:
                batch_X = X[step * batch_size:(step + 1) * batch_size]
            batch_output = self.forward(batch_X)
            output.append(batch_output)
        return np.vstack(output)


    def forward(self, X):
        self.input_layer.forward(X)
        for layer in self.layers:
            layer.forward(layer.prev.output)
        return layer.output


    def backward(self, output, y):
        self.loss.backward(output, y)
        for layer in reversed(self.layers):
            layer.backward(layer.next.dinputs)


    def summary(self):
        print("_"*100)
        print("{:<15} {:<15} {:<20} {:<20} {:<15} {:<15}".format("Type", "Name", "Input Shape", "Output Shape", "Activation", "Param #"))
        print("="*100)
        trainable_param = 0
        for id_layer, layer in enumerate(self.layers):
            type_layer = str(type(layer).__name__)
            name_layer = str(layer.name)
            input_shape_layer = str(layer.input_shape)
            if len(layer.input_shape) > 2:
                __input_shape = list(layer.input_shape)
                __input_shape[0] = None
                input_shape_layer = str(tuple(__input_shape))
            else:
                __input_shape = list(layer.input_shape)
                __input_shape = [None] + __input_shape
                input_shape_layer = str(tuple(__input_shape))
            output_shape_layer = str(layer.output_shape)
            if len(layer.output_shape) > 2:
                __output_shape = list(layer.output_shape)
                __output_shape[0] = None
                output_shape_layer = str(tuple(__output_shape))
            else:
                __output_shape = list(layer.output_shape)
                __output_shape = [None] + __output_shape
                output_shape_layer = str(tuple(__output_shape))
            if id_layer == len(self.layers) - 1:
                output_shape_layer = str((None, 1, layer.units))
            activation_layer = str(type(layer.activation).__name__)
            if activation_layer == "NoneType":
                activation_layer = "None"
            param_layer = 0
            if hasattr(layer, 'weights'):
                param_layer += np.prod(layer.weights.shape)
            if hasattr(layer, 'biases'):
                param_layer += np.prod(layer.biases.shape)
            trainable_param += param_layer
            param_layer = str(param_layer)
            print("{:<15} {:<15} {:<20} {:<20} {:<15} {:<15}\n".format(type_layer, name_layer, input_shape_layer, output_shape_layer, activation_layer, param_layer))
        print("="*100)
        print("Total params: {:,}".format(trainable_param))
        print("Trainable params: {:,}".format(trainable_param))
        print("Non-trainable params: 0")
        print("_"*100)


    def save(self, path):
        model = copy.deepcopy(self)
        model.loss.new_pass()
        model.accuracy.new_pass()

        model.input_layer.__dict__.pop('output', None)
        model.loss.__dict__.pop('dinputs', None)

        for layer in model.layers:
            for property in ['inputs', 'output', 'dinputs',
                             'dweights', 'dbiases']:
                layer.__dict__.pop(property, None)

        with open(path, 'wb') as f:
            pickle.dump(model, f)


    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            model = pickle.load(f)
        return model