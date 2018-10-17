#!/usr/bin/env python
# encoding: utf-8

import tensorflow as tf

slim = tf.contrib.slim


def res_arg_scope(is_training, weight_decay=0.0005):
    with slim.arg_scope(
        [slim.conv2d],
            activation_fn=tf.nn.relu,
            weights_regularizer=slim.l2_regularizer(weight_decay),
            biases_initializer=tf.zeros_initializer(),
            normalizer_fn=slim.batch_norm,
            normalizer_params={'is_training': is_training}):
        with slim.arg_scope(
                [slim.conv2d, slim.max_pool2d], padding='SAME') as arg_sc:
            return arg_sc


def bottleneckv1(inputs, filters, scope=None):
    nb_filters1, nb_filters2, nb_filters3 = filters
    with tf.variable_scope(scope, 'bottleneckv1'):
        # 1*1
        branch2a = slim.conv2d(inputs, nb_filters1, (1, 1), scope='branch2a')
        # 3*3
        branch2b = slim.conv2d(branch2a, nb_filters2, (3, 3), scope='branch2b')
        # 1*1
        branch2c = slim.conv2d(
            branch2b,
            nb_filters3, (1, 1),
            scope='branch2c',
            activation_fn=None)

        # skip connected way
        x = tf.add(branch2c, inputs)
        x = tf.nn.relu(x)
        return x


def bottleneckv2(inputs, filters, version='reduce', scope=None):
    # two versions of inception model:
    # 1. navie version
    # 2. reduce version
    if version == 'reduce':
        reduce_on = True
        assert len(filters) == 6
        nb_filters1, nb_filters2, nb_filters3, nb_filters4, nb_filters5, nb_filters6 = filters
    else:
        reduce_on = False
        assert len(filters) == 3
        nb_filters1, nb_filters3, nb_filters5 = filters

    with tf.variable_scope(scope, 'bottleneckv2'):
        branch1 = slim.conv2d(inputs, nb_filters1, (1, 1), scope='branch1')

        if reduce_on:
            branch2a = slim.conv2d(inputs, nb_filters2,
                                   (1, 1), scope='branch2a')
        branch2b = slim.conv2d(branch2a, nb_filters3, (3, 3), scope='branch2b')

        if reduce_on:
            branch3a = slim.conv2d(inputs, nb_filters4,
                                   (1, 1), scope='branch3a')
        branch3b = slim.conv2d(branch3a, nb_filters5, (5, 5), scope='branch3b')

        branch4a = slim.max_pool2d(inputs, (3, 3), scope='branch4a')
        if reduce_on:
            branch4b = slim.conv2d(branch4a, nb_filters6,
                                   (1, 1), scope='branch4b')

        return tf.concat([branch1, branch2b, branch3b, branch4b], axis=3)


def bottleneck(inputs, filters, version, scope=None):
    if version == 'identity':
        # filters = [10, 10, 20]
        return bottleneckv1(inputs, filters, scope=scope)
    else:
        # if version == 'reduce':
            # filters = [10, 5, 10, 5, 10, 10]
        # elif version == 'navie':
            # filters = [10, 10, 10]
        return bottleneckv2(inputs, filters, version=version, scope=scope)


def resnet(inputs, version, is_training=True, weight_decay=0.0005):
    with slim.arg_scope(res_arg_scope(is_training, weight_decay=weight_decay)):
        with tf.variable_scope('resnetv1'):
            x = slim.conv2d(inputs, 20, (3, 3), scope='conv1')
            x = slim.max_pool2d(x, (3, 3), 2, scope='pool1')

            x = bottleneck(x, [10, 10, 20], version, scope='a')

            x = slim.conv2d(x, 32, (3, 3), scope='conv2')
            x = slim.max_pool2d(x, (3, 3), 2, scope='pool2')

            x = bottleneck(x, [16, 16, 32], version, scope='b')

            #
            x = tf.reduce_max(x, [1, 2], name='pool3')

            x = slim.softmax(x, scope='softmax')
            return x
