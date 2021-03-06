import numpy as np
import pickle
import copy
from .layers import Input, Dense
from .activation import Softmax, ReLU

class Model:

    def __init__(self):
        self.layers = []


    def add(self, layer):
        if isinstance(layer, Dense):
            self.layers.append(layer)
            if layer.activation != None:
                self.layers.append(layer.activation)
        else:
            self.layers.append(layer)


    def compile(self, loss, optimizer, accuracy):
        if loss != None:
            self.loss = loss
        if optimizer != None:
            self.optimizer = optimizer
        if accuracy != None:
            self.accuracy = accuracy

        self.input_layer = Input()
        layer_count = len(self.layers)
        self.trainable_layers = []
        for i in range(layer_count):
            if i == 0:
                self.layers[i].prev = self.input_layer
                self.layers[i].next = self.layers[i + 1]
            elif i < layer_count - 1:
                self.layers[i].prev = self.layers[i - 1]
                self.layers[i].next = self.layers[i + 1]
            else:
                self.layers[i].prev = self.layers[i - 1]
                self.layers[i].next = self.loss
                self.output_layer_activation = self.layers[i]

            if hasattr(self.layers[i], 'weights'):
                self.trainable_layers.append(self.layers[i])
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
                if isinstance(self.layers[-1], Softmax):
                    predictions = np.argmax(predictions, axis=1)
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