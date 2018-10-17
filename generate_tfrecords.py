#!/usr/bin/env python
# encoding: utf-8

import tensorflow as tf
import os
import numpy as np
import threading
from image_coder import ImageCoder

tf.app.flags.DEFINE_integer(
    'num_shards', 2, 'Number of shards in TFRecord file')
tf.app.flags.DEFINE_integer(
    'num_threads', 2, 'Number of threads to preprocess the images')
tf.app.flags.DEFINE_string('data_dir', '/tmp/', 'Training data directory')
tf.app.flags.DEFINE_string('output_dir', '/tmp/', 'Output data directory')


FLAGS = tf.app.flags.FLAGS


DIRECTIONS = [(0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1,
                                                                           -1)]


def nextpos(oldpos, direction):
    newpos = []
    delta = DIRECTIONS[direction]
    newpos.append(oldpos[0] + delta[0])
    newpos.append(oldpos[1] + delta[1])
    return newpos


def _bytes_feature(value):
    return tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[tf.compat.as_bytes(value)]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _convert_to_example(image_buffer, im_shape, im_fn, label, text):
    """
    Args:
        image_buffer: jpeg encoded string(bytes)
    """
    # store image buffer encoded with jpeg
    image_format = 'jpeg'
    channel = 3

    example = tf.train.Example(
        features=tf.train.Features(
            feature={
                'image/encoded': _bytes_feature(image_buffer),
                'image/height': _int64_feature(im_shape[0]),
                'image/width': _int64_feature(im_shape[1]),
                'image/format': _bytes_feature(image_format),
                'image/channel': _int64_feature(channel),
                'image/class/label': _int64_feature(label),
                'image/class/text': _bytes_feature(text)
            }))
    return example


label_map = {'normal': 0, 'cancer': 1}
classes = ['normal', 'cancer']


def labelmap(label_text):
    return label_map[label_text]


def label2text(label):
    """
    convert integer label to class text
    """
    return classes[label]


def is_jpeg(filename):
    if os.path.splitext(filename)[-1] in ['.jpg', '.jpeg']:
        return True
    return False


def _find_image_files(data_rootpath):
    normal_filenames = []
    normal_label = 0
    cancer_filenames = []
    cancer_label = 1
    classes = ['normal', 'cancer']
    for class_text in classes:
        filenames_list = normal_filenames if class_text == 'normal' else cancer_filenames
        tmp_path = os.path.join(data_rootpath, class_text)
        for dirpath, dirnames, img_fns in os.walk(tmp_path):
            for img_fn in img_fns:
                if is_jpeg(img_fn):
                    filenames_list.append(os.path.join(dirpath, img_fn))
    normal_labels = [normal_label] * len(normal_filenames)
    cancer_labels = [cancer_label] * len(cancer_filenames)
    all_filenames = normal_filenames + cancer_filenames
    all_labels = normal_labels + cancer_labels
    all_filenames = np.asarray(all_filenames)
    all_labels = np.asarray(all_labels)

    # shuffle idx
    idx = np.arange(all_labels.size)
    np.random.shuffle(idx)

    return all_filenames[idx], all_labels[idx]


def crop_image(boundary_points, im, x_pad, y_pad):
    # find the max and min of points
    ymin, xmin = im.shape[:2]
    ymax, xmax = -1, -1
    for point in boundary_points:
        if xmin > point[0]:
            xmin = point[0]
        if ymin > point[1]:
            ymin = point[1]
        if xmax < point[0]:
            xmax = point[0]
        if ymax < point[1]:
            ymax = point[1]

    xmin -= x_pad
    ymin -= y_pad
    xmin = 0 if xmin < 0 else xmin
    xmax += x_pad
    ymax += y_pad
    xmax = im.shape[1] if xmax >= im.shape[1] else xmax
    ymax = im.shape[0] if ymax >= im.shape[0] else ymax

    print("Crop Area: (ymin:{},xmin:{},ymax:{},xmin:{})".format(
        ymin, xmin, ymax, xmax))

    return im[ymin:ymax + 1, xmin:xmax + 1]


def generate_points(start_pos, directions):
    points = []
    cur_pos = start_pos
    points.append(cur_pos)

    for direction in directions:
        cur_pos = nextpos(cur_pos, direction)
        points.append(cur_pos)
    return np.asarray(points, np.int32)


def generate_boundary_points(overlay_fn):
    # extract chain data from overlay file
    with open(overlay_fn, 'r') as f:
        for line in f.readlines():
            line = line.strip().split(' ')
            if not line[0].isdigit():
                continue
            # convert str to int
            line = [int(item) for item in line[:-1]]

            start_x, start_y = line[:2]
            boundary = line[2:]
            break

    # generate bounary points
    boundary_points = generate_points((start_x, start_y), boundary)
    return boundary_points


def _process_image(img_fn, overlay_fn, coder, encoding='jpeg'):
    # how to process image for storing
    with open(img_fn, 'rb') as f:
        image_data = f.read()

    # decode image for get image shape and crop operation
    decode_image_data = coder.decode_jpeg(image_data)

    if os.path.exists(overlay_fn):
        print("Using Crop Operation!")
        # crop pipeline
        boundary_points = generate_boundary_points(overlay_fn)

        croped_image = crop_image(boundary_points, decode_image_data, 100, 100)
        image_data = coder.encode_jpeg(croped_image)

    shape = decode_image_data.shape

    return image_data, shape


def split_works(num_data, num_split):
    """split_works
    Args:
    :param num_data: number of elements
    :param num_split: number of splited works

    Returns:
    ranges: list, num_split in all,contains idx of data elements
    """
    idx_list = np.arange(num_data)
    spacing = np.linspace(0, num_data, num_split + 1).astype(np.int)
    ranges = []
    for idx in range(num_split):
        ranges.append(idx_list[spacing[idx]:spacing[idx + 1]])
    return ranges


def generate_overlay_fn(img_path):
    prefix, suffix = os.path.splitext(img_path)
    return prefix + '.OVERLAY'


def _process_images_batch(thread_idx, shards_idxs, dataset_name, num_shards, filenames, labels, coder, output_dir):
    split_idxs_per_thread = split_works(len(filenames), len(shards_idxs))
    for idx, shard_idx in enumerate(shards_idxs):
        option = tf.python_io.TFRecordOptions(
            tf.python_io.TFRecordCompressionType.GZIP)
        output_filename = '%s-%.5d-of-%.5d.tfrecord' % (
            dataset_name, shard_idx, num_shards)
        output_file = os.path.join(output_dir, output_filename)
        with tf.python_io.TFRecordWriter(output_file, option) as writer:
            for filename, label in zip(filenames[split_idxs_per_thread[idx]], labels[split_idxs_per_thread[idx]]):
                overlay_fn = generate_overlay_fn(filename)
                image_data, im_shape = _process_image(
                    filename, overlay_fn, coder)
                example = _convert_to_example(
                    image_data, im_shape, filename, label, label2text(label))
                writer.write(example.SerializeToString())


def _process_images_all(all_filenames, all_labels, dataset_name, all_num_threads, all_num_shards, output_dir):

    # split works first by using idx
    all_num = len(all_filenames)
    split_idxs = split_works(all_num, all_num_threads)

    all_threads = []
    # tfrecords file that should be assigned for each thread
    # num_shards_per_batch = all_num_shards / all_num_threads
    shards_split_idxs = split_works(all_num_shards, all_num_threads)
    coord = tf.train.Coordinator()

    coder = ImageCoder()
    # for each thread do some work accoding to their work list
    for thread_idx in range(all_num_threads):
        # images and label pairs
        filenames = all_filenames[split_idxs[thread_idx]]
        labels = all_labels[split_idxs[thread_idx]]
        shards_idxs = shards_split_idxs[thread_idx]
        # threads
        args = (thread_idx, shards_idxs,
                dataset_name, all_num_shards, filenames, labels, coder, output_dir)
        thread = threading.Thread(target=_process_images_batch, args=args)
        thread.start()

        # collect threads
        all_threads.append(thread)

    # coord to reduce
    coord.request_stop()
    coord.join(all_threads)


def main(unused_argv):
    all_filenames, all_labels = _find_image_files(FLAGS.data_dir)
    _process_images_all(all_filenames, all_labels, 'ddsm',
                        FLAGS.num_threads, FLAGS.num_shards, FLAGS.output_dir)


if __name__ == '__main__':
    tf.app.run()
