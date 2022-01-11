import numpy as np


class Adam:

    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7, beta_1=0.9, beta_2=0.999):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2


    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate *  (1. / (1. + self.decay * self.iterations))


    def update_params(self, layer):
        caches = layer.caches
        caches["weight_momentums"] = self.beta_1 * caches["weight_momentums"] + (1 - self.beta_1) * layer.dweights
        caches["bias_momentums"] = self.beta_1 * caches["bias_momentums"] + (1 - self.beta_1) * layer.dbiases

        weight_momentums_corrected = caches["weight_momentums"] / (1 - self.beta_1 ** (self.iterations + 1))
        bias_momentums_corrected = caches["bias_momentums"] / (1 - self.beta_1 ** (self.iterations + 1))

        caches["weight_cache"] = self.beta_2 * caches["weight_cache"] + (1 - self.beta_2) * layer.dweights**2
        caches["bias_cache"] = self.beta_2 * caches["bias_cache"] + (1 - self.beta_2) * layer.dbiases**2

        weight_cache_corrected = caches["weight_cache"] / (1 - self.beta_2 ** (self.iterations + 1))
        bias_cache_corrected = caches["bias_cache"] / (1 - self.beta_2 ** (self.iterations + 1))

        layer.weights += -self.current_learning_rate * weight_momentums_corrected / (np.sqrt(weight_cache_corrected) + self.epsilon)
        layer.biases += -self.current_learning_rate * bias_momentums_corrected / (np.sqrt(bias_cache_corrected) + self.epsilon)

        layer.caches = caches

    def post_update_params(self):
        self.iterations += 1