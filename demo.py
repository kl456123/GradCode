#!/usr/bin/env python
# encoding: utf-8

"""
predict class when given some images
here it is not necessary to make mini batch
"""

import tensorflow as tf
import numpy as np
import cv2
import cifar10

from resnet50 import resnet

tf.app.flags.DEFINE_string('img_path', '', """path of image for prediction""")
tf.app.flags.DEFINE_string(
    'checkpoint_dir', '', """directory of checkpoints""")

FLAGS = tf.app.flags.FLAGS


def decode_py(img_path):
    return cv2.imread(img_path)


def decode_tf(img_path):
    pass


def main(unused_argc):
    img = decode_py(FLAGS.img_path)

    img_pl = tf.placeholder(shape=[None, None, None, 3], dtype=tf.float32)

    logit = resnet(img_pl, 'identity')
    ema = tf.train.ExponentialMovingAverage(cifar10.MOVING_AVERAGE_DECAY)
    variables_to_restore = ema.variables_to_restore()

    saver = tf.train.Saver(variables_to_restore)

    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)

        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        logit_value = sess.run(logit, feed_dict={img_pl: img})

    label = np.argmax(logit_value, axis=0)
    print(label)


if __name__ == '__main__':
    tf.app.run()
