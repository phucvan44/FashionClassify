import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt

from lib.accuracy import  Accuracy
from lib.activation import ReLU, Softmax
from lib.layers import Dense, Flatten, MaxPooling2D, Conv2D
from lib.loss import CategoricalCrossentropy
from lib.model import Model
from lib.optimizer import Adam


def main():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    fashion_mnist = tf.keras.datasets.fashion_mnist

    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    train_images = train_images / 255.0
    test_images = test_images / 255.0

    model = Model()

    filters = 3
    kernel_size = 3
    pool_size = 3

    # model.add(Conv2D(filters, input_shape=(1, 28, 28), kernel_size=kernel_size, activation=ReLU()))
    # model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Flatten(input_shape=(1, 28, 28)))
    model.add(Dense(128, name="Layer1", activation=ReLU()))
    model.add(Dense(64, name="Layer2", activation=ReLU()))
    model.add(Dense(10, name="Layer3", activation=Softmax()))

    model.compile(
        loss=CategoricalCrossentropy(),
        optimizer=Adam(decay=1e-3),
        accuracy=Accuracy()
    )

    model.summary()
    model.fit(train_images, train_labels, epochs=10, batch_size=32)
    model.save("backup")


if __name__ == "__main__":
    main()