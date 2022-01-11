import numpy as np


class Accuracy:


    def compare(self, predictions, y):
        return predictions == y


    def calculate(self, predictions, y):
        comparisons = self.compare(predictions, y)
        accuracy = np.mean(comparisons)
        self.accumulated_sum += np.sum(comparisons)
        self.accumulated_count += len(np.array(comparisons))
        return accuracy


    def calculate_accumulated(self):
        accuracy = self.accumulated_sum / self.accumulated_count
        return accuracy


    def new_pass(self):
        self.accumulated_sum = 0
        self.accumulated_count = 0



