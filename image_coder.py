#!/usr/bin/env python
# encoding: utf-8

import tensorflow as tf


class ImageCoder(object):
    def __init__(self):
        self._sess = tf.Session()

        # convert bytes to RGB 3D arr
        self._decode_jpeg_data_pl = tf.placeholder(dtype=tf.string)
        self._decode_jpeg = tf.image.decode_jpeg(
            self._decode_jpeg_data_pl, channels=3)

        self._decode_jpeg_pl = tf.placeholder(
            shape=(None, None, 3), dtype=tf.uint8)

        self._decode_jpeg_data = tf.image.encode_jpeg(
            self._decode_jpeg_pl, format='rgb', quality=100)

    def decode_jpeg(self, image_data):
        image = self._sess.run(self._decode_jpeg, feed_dict={
            self._decode_jpeg_data_pl: image_data})
        assert image.shape[2] == 3
        return image

    def encode_jpeg(self, decoded_jpeg):
        image_data = self._sess.run(self._decode_jpeg_data, feed_dict={
            self._decode_jpeg_pl: decoded_jpeg
        })
        return image_data
