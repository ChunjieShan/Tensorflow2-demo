#!/usr/bin/python3
# -*- coding: utf8 -*-

import tensorflow as tf


class Simpleconv3(tf.keras.Model):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        super(Simpleconv3, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(32, (2, 2),
                                            strides=2,
                                            activation='relu')
        self.conv2 = tf.keras.layers.Conv2D(64, (2, 2),
                                            strides=2,
                                            activation='relu')
        self.conv3 = tf.keras.layers.Conv2D(64, (2, 2),
                                            strides=2,
                                            activation='relu')
        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(256, activation='relu')
        self.fc2 = tf.keras.layers.Dense(self.num_classes,
                                         activation='softmax')

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


if __name__ == "__main__":
    import tensorflow as tf
    import numpy as np
    image = np.random.randn(1, 48, 48, 3)
    model = Simpleconv3()
    output = model(image)
    print(output)
