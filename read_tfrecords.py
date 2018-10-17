#!/usr/bin/env python
# encoding: utf-8

"""
    using tf.data api to construct ETL input pipeline,
preprocess dataset,data augmentation,and then return
batches(images and labels pair)


# some points for performance
1. fuse crop and decode for jpeg image
this menthod is not adopted due to code will be coupled with each other
2. the order of transformation process for dataset
3. parallelize data transformation
"""

import tensorflow as tf
import os
import numpy as np

tf.app.flags.DEFINE_string(
    'data_dir', '/tmp', 'data dir of tfrecords data files')
tf.app.flags.DEFINE_integer('image_size', 300, 'resize image to square')
tf.app.flags.DEFINE_integer(
    'num_parallel', 8, 'set num_parallel_calls for data transformation')
tf.app.flags.DEFINE_integer('num_epoches', 10, 'num epoches to train')

tf.app.flags.DEFINE_integer('num_examples_prefetch',
                            10, 'num examples to prefetch')
tf.app.flags.DEFINE_integer('num_batches', 10, 'num example in a batch')

FLAGS = tf.app.flags.FLAGS


def _is_tfrecords(filename):
    if os.path.splitext(filename)[-1] == '.tfrecord':
        return True
    return False


def generate_tfrecords(tfrecords_dir):
    all_file_path = []
    for filename in os.listdir(tfrecords_dir):
        if _is_tfrecords(filename):
            all_file_path.append(os.path.join(tfrecords_dir, filename))
    return all_file_path


def _decode(example_proto):
    features = {'image/encoded': tf.FixedLenFeature((), tf.string),
                'image/height': tf.FixedLenFeature((), tf.int64),
                'image/width': tf.FixedLenFeature((), tf.int64),
                'image/format': tf.FixedLenFeature((), tf.string),
                'image/channel': tf.FixedLenFeature((), tf.int64),
                'image/class/label': tf.FixedLenFeature((), tf.int64),
                'image/class/text': tf.FixedLenFeature((), tf.string)
                }

    # TODO
    # add object bbox infomation to used for some augmentation ops(e,g crop etc)
    # of course modify generate_tfrecords at the same time

    parsed_features = tf.parse_single_example(example_proto, features=features)

    # decode from jpeg data to uint8
    decode_image_data = tf.image.decode_jpeg(
        parsed_features['image/encoded'], channels=3)

    # uint8 to float32
    # some rescaled ops are done to make data in range from 0 to 1
    image = tf.image.convert_image_dtype(decode_image_data, tf.float32)

    # cast from tf.int64 to tf.int32
    label = tf.cast(parsed_features['image/class/label'], tf.int32)

    # shape is no needed for encoded data by jpeg(or other image type)
    # channel = tf.cast(parsed_features['image/channel',tf.int32])
    # height = tf.cast(parsed_features['image/height',tf.int32])
    # width = tf.cast(parsed_features['image/width',tf.int32])
    return image, label


def _distort_color(image, color_ordering, scope=None):
    """Distort the color of the image.

    Each color distortion is non-commutative and thus ordering of the color ops
    matters. Ideally we would randomly permute the ordering of the color ops.
    Rather than adding that level of complication, we select a distinct ordering
    of color ops for each preprocessing thread.

    Args:
      image: Tensor containing single image.
      thread_id: preprocessing thread ID.
      scope: Optional scope for name_scope.
    Returns:
      color-distorted image
    """
    with tf.name_scope(values=[image], name=scope, default_name='distort_color'):

        if color_ordering == 0:
            image = tf.image.random_brightness(image, max_delta=32. / 255.)
            image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            image = tf.image.random_hue(image, max_delta=0.2)
            image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        elif color_ordering == 1:
            image = tf.image.random_brightness(image, max_delta=32. / 255.)
            image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
            image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            image = tf.image.random_hue(image, max_delta=0.2)

        # The random_* ops do not necessarily clamp.
        image = tf.clip_by_value(image, 0.0, 1.0)
        return image


def _preprocess():
    pass


def _postprocess(image, label):
    # do normalization like inception code
    # maybe some problem
    image = tf.subtract(image, 0.5)
    image = tf.multiply(image, 2.0)
    return image, label


def _augment(image, label):
    # do crop op first
    # due to lack of bbox info,so assmue bbox is the entire image
    # note that all param is setted according to inception code
    # im_shape = tf.shape(image)
    # ymax = im_shape[0]-1
    # xmax = im_shape[1]-1
    sampled_regions = tf.image.sample_distorted_bounding_box(
        tf.shape(image),
        bounding_boxes=np.empty((1, 0, 4)),
        min_object_covered=0.5,
        aspect_ratio_range=[0.75, 1.33],
        area_range=[0.05, 1.0],
        max_attempts=100,
        use_image_if_no_bounding_boxes=True
    )
    # a single bbox
    # note that distorted_box is like [ymin,xmin,ymax,xmax]
    bbox_begin, bbox_size, distorted_bbox = sampled_regions

    # crop region to get 'new image'
    distorted_image = tf.slice(image, bbox_begin, bbox_size)

    # then resize it to the same size
    # select a resize method randomly
    resize_method = np.random.randint(4, size=1)[0]

    distorted_image = tf.image.resize_images(
        distorted_image, [FLAGS.image_size, FLAGS.image_size], method=resize_method)

    # restore image shape
    distorted_image.set_shape([FLAGS.image_size, FLAGS.image_size, 3])

    # distort color here
    # 2 kinds of color_ordering
    color_ordering = np.random.randint(2, size=1)[0]
    distorted_image = _distort_color(distorted_image, color_ordering)

    # flip
    image = tf.image.random_flip_left_right(distorted_image)
    return image, label


def read_tfrecords_batch(tfrecords_dir, num_epoches=10, num_examples_prefetch=3, num_batches=10, num_parallel=8, mode='train'):
    all_tfrecords = generate_tfrecords(tfrecords_dir)

    dataset = tf.data.TFRecordDataset(all_tfrecords, compression_type='GZIP')

    # do something ops in order

    dataset = dataset.map(_decode, num_parallel_calls=num_parallel)

    dataset = dataset.map(_augment, num_parallel_calls=num_parallel)

    dataset = dataset.map(_postprocess, num_parallel_calls=num_parallel)

    # buffer size to control the degree of random
    # if buffer size is the number of the dataset, random is completely

    # combined with each other
    # dataset = dataset.shuffle(num_batches * 3 + 1000)
    # dataset = dataset.repeat(num_epoches)
    dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(
        num_batches * 3, num_epoches))

    dataset = dataset.batch(num_batches)

    dataset = dataset.prefetch(num_examples_prefetch)

    if mode == 'train':
        iterator = dataset.make_one_shot_iterator()
        return iterator.get_next()
    else:
        iterator = dataset.make_initializable_iterator()
        # for reinitialization
        return iterator.get_next(), iterator


def visualize(images):
    import matplotlib.pyplot as plt
    for img_idx in range(images.shape[0]):
        plt.imshow(images[img_idx])
        plt.show()


def main(unused_argc):
    # some dataset config

    image_batch, label_batch = read_tfrecords_batch(
        FLAGS.data_dir, FLAGS.num_epoches, FLAGS.num_examples_prefetch, FLAGS.num_batches, num_parallel=FLAGS.num_parallel)
    step = 0
    with tf.Session() as sess:
        # mybe do some init ops for global variable and local variable first here
        try:
            while True:
                images, label = sess.run([image_batch, label_batch])
                step += 1
                if label == 1:
                    visualize(images)
                    print('label of image: ', label)
        except tf.errors.OutOfRangeError:
            print("All is done!")


if __name__ == '__main__':
    tf.app.run()
