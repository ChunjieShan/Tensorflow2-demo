#!/usr/bin/python3
# -*- coding: utf8 -*-

import tensorflow as tf

txt_file = "./train_shuffle.txt"
batch_size = 64
num_classes = 4
image_size = (48, 48)


class ImageData():
    def read_txt_file(self):
        self.img_paths = []
        self.labels = []
        for line in open(self.txt_file, 'r'):
            line = line.splitlines()
            new_line = line[0].split(' ')
            self.img_paths.append(new_line[0])
            self.labels.append(int(new_line[1]))

    def __init__(self,
                 txt_file,
                 batch_size,
                 num_classes,
                 image_size,
                 buffer_scale=100):
        self.txt_file = txt_file
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.image_size = image_size
        buffer_size = buffer_scale * batch_size

        self.read_txt_file()
        self.dataset_size = len(self.labels)
        print("The length of datset is: ", self.dataset_size)

        self.img_paths = tf.convert_to_tensor(self.img_paths, dtype=tf.string)
        self.labels = tf.convert_to_tensor(self.labels, dtype=tf.int32)

        data = tf.data.Dataset.from_tensor_slices(
            (self.img_paths, self.labels))
        print("Data type: ", type(data))

        data = data.map(self.parse_function)
        data = data.repeat(10)
        data = data.shuffle(buffer_size=buffer_size)

        self.data = data.batch(batch_size)
        print("self.data type = ", type(self.data))

    def augment_dataset(self, image, size):
        distorted_image = tf.image.random_brightness(image, max_delta=63)
        distorted_image = tf.image.random_contrast(distorted_image,
                                                   lower=0.2,
                                                   upper=1.8)
        float_image = tf.image.per_image_standardization(distorted_image)
        return float_image

    def parse_function(self, file_name, label):
        label_ = tf.one_hot(label, self.num_classes)
        img = tf.io.read_file(file_name)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.convert_image_dtype(img, dtype=tf.float32)
        img = tf.image.random_crop(img,
                                   [self.image_size[0], self.image_size[1], 3])
        img = tf.image.random_flip_left_right(img)
        img = self.augment_dataset(img, self.image_size)

        return img, label_


if __name__ == "__main__":
    from net import Simpleconv3
    dataset = ImageData(txt_file, batch_size, num_classes, image_size)
