import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt

from lib.accuracy import  Accuracy
from lib.activation import ReLU, Softmax
from lib.layers import Dense, Flatten, Conv2D, MaxPooling2D
from lib.loss import CategoricalCrossentropy
from lib.model import Model
from lib.optimizer import Adam


def main():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    fashion_mnist = tf.keras.datasets.fashion_mnist

    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    train_images = train_images / 255.0
    test_images = test_images / 255.0

    model = Model()
    #
    model.add(Conv2D(filters=32, kernel_size=3, input_shape=(1, 28, 28), name="conv layer 1", activation=ReLU()))
    model.add(MaxPooling2D(pool_size=3, name="maxpool layer 1"))
    model.add(Flatten())
    model.add(Dense(units = 128, name="Layer 1", activation=ReLU()))
    model.add(Dense(units=10, name="Layer 2", activation=Softmax()))
    #
    model.compile(
        loss=CategoricalCrossentropy(),
        optimizer=Adam(decay=1e-3),
        accuracy=Accuracy()
    )
    model.summary()
    model.fit(train_images, train_labels, epochs=5, batch_size=32)
    # #
    model.save("backup")

if __name__ == "__main__":
    main()