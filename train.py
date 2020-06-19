#!/usr/bin/python3
# -*- coding: utf8 -*-

import tensorflow as tf
from net import Simpleconv3
from datasets import ImageData


class Train_model():
    def __init__(self, data, model, criterion, optimizer, epochs):
        self.data = data
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.epochs = epochs

    def train(self):
        self.model.compile(loss=self.criterion,
                           optimizer=self.optimizer,
                           metrics=['accuracy'])
        self.model.fit(self.data, epochs=self.epochs)
        self.model.save("./")


if __name__ == "__main__":
    txt_file = "./train_shuffle.txt"
    batch_size = 50
    num_classes = 4
    image_size = (48, 48)
    learning_rate = 1e-3
    epochs = 2
    num_classes = 2
    criterion = "categorical_crossentropy"
    optimizer = tf.keras.optimizers.Adam()

    model = Simpleconv3(num_classes=2)
    dataset = ImageData(txt_file, batch_size, num_classes, image_size)
    history_ht = Train_model(dataset.data, model, criterion, optimizer, epochs)
    history_ht.train()
