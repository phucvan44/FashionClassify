import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt

from lib.accuracy import  Accuracy
from lib.activation import ReLU, Softmax
from lib.layers import Dense
from lib.loss import CategoricalCrossentropy
from lib.model import Model
from lib.optimizer import Adam


def main():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    fashion_mnist = tf.keras.datasets.fashion_mnist

    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    _train_images, _test_images = [train_images.copy(), test_images.copy()]

    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    train_images = train_images / 255.0
    train_images = train_images.reshape(train_images.shape[0], -1).astype(np.float32)
    test_images = test_images / 255.0
    test_images = test_images.reshape(test_images.shape[0], -1).astype(np.float32)

    model = Model()

    model.add(Dense(shape=(train_images.shape[1], 128), name="Layer1", activation=ReLU()))
    model.add(Dense(shape=(128, 64), name="Layer2", activation=ReLU()))
    model.add(Dense(shape=(64, 10), name="Layer3", activation=Softmax()))

    model.compile(
        loss=CategoricalCrossentropy(),
        optimizer=Adam(decay=1e-3),
        accuracy=Accuracy()
    )
    model.fit(train_images, train_labels, epochs=3, batch_size=32)

    model.save("backup")

if __name__ == "__main__":
    main()