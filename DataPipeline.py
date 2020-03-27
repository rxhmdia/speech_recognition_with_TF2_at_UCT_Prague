import os

import numpy as np
import tensorflow as tf

from FLAGS import FLAGS


def _parse_proto(example_proto):
    features = {
        'x': tf.io.FixedLenSequenceFeature([FLAGS.num_features], tf.float32, allow_missing=True),
        'y': tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
    }
    parsed_features = tf.io.parse_single_example(example_proto, features)
    return parsed_features['x'], parsed_features['y']


def _read_tfrecords(file_names=("file1.tfrecord", "file2.tfrecord", "file3.tfrecord"),
                    shuffle=False, seed=None, block_length=FLAGS.num_train_data, cycle_length=8):
    files = tf.data.Dataset.list_files(file_names, shuffle=shuffle, seed=seed)
    ds = files.interleave(lambda x: tf.data.TFRecordDataset(x).map(_parse_proto,
                                                                   num_parallel_calls=FLAGS.num_cpu_cores),
                          block_length=block_length,
                          cycle_length=cycle_length,
                          num_parallel_calls=FLAGS.num_cpu_cores)
    ds = ds.map(lambda x, y: (x, y, tf.shape(x)[0], tf.size(y)), num_parallel_calls=FLAGS.num_cpu_cores)
    return ds


def _bucket_and_batch(ds, bucket_boundaries):
    num_buckets = len(bucket_boundaries) + 1
    bucket_batch_sizes = [FLAGS.batch_size_per_GPU] * num_buckets
    padded_shapes = (tf.TensorShape([None, FLAGS.num_features]),  # cepstra padded to maximum time in batch
                     tf.TensorShape([None]),  # labels padded to maximum length in batch
                     tf.TensorShape([]),  # sizes not padded
                     tf.TensorShape([]))  # sizes not padded
    padding_values = (tf.constant(FLAGS.feature_pad_val, dtype=tf.float32),  # cepstra padded with feature_pad_val
                      tf.constant(FLAGS.label_pad_val, dtype=tf.int64),  # labels padded with label_pad_val
                      0,  # size(cepstrum) -- unused
                      0)  # size(label) -- unused

    bucket_transformation = tf.data.experimental.bucket_by_sequence_length(
        element_length_func=lambda x, y, size_x, size_y: size_x,
        bucket_boundaries=bucket_boundaries,
        bucket_batch_sizes=bucket_batch_sizes,
        padded_shapes=padded_shapes,
        padding_values=padding_values
    )

    ds = ds.apply(bucket_transformation)
    return ds


def load_datasets(load_dir):
    path_gen = os.walk(load_dir)

    ds_train = None
    ds_test = None

    # load datasets from .tfrecord files in test and train folders
    for path, subfolders, files in path_gen:
        folder_name = os.path.split(path)[-1]
        files = [f for f in files if '.tfrecord' in f]
        fullpaths = [os.path.join(path, f) for f in files]
        if folder_name == '' and len(files) > 0:
            num_data = FLAGS.num_train_data + FLAGS.num_test_data
            ds = _read_tfrecords(fullpaths, shuffle=True, seed=FLAGS.shuffle_seed, block_length=num_data,
                                 cycle_length=1)
            ds = ds.shuffle(FLAGS.num_train_data + FLAGS.num_test_data, seed=FLAGS.shuffle_seed,
                            reshuffle_each_iteration=False)
            ds_train = ds.take(FLAGS.num_train_data)
            ds_test = ds.skip(FLAGS.num_train_data)
            print('joined dataset loaded from {} and '
                  'split into ds_train ({}) and ds_test ({})'.format(path, FLAGS.num_train_data, 'rest'))
            break
        if folder_name == 'test':
            ds_test = _read_tfrecords(fullpaths, block_length=FLAGS.num_test_data)
            print('test dataset loaded from {}'.format(path))
        elif folder_name == 'train':
            # don't shuffle if using shards, because bucketting doesn't work well with shards
            ds_train = _read_tfrecords(fullpaths, block_length=FLAGS.num_train_data)
            print('train dataset loaded from {}'.format(path))
        else:
            continue

    # BUCKET AND BATCH DATASET
    bucket_boundaries = list(range(FLAGS.min_time, FLAGS.max_time + 1, FLAGS.bucket_width))
    num_buckets = len(bucket_boundaries) + 1
    num_train_batches = (np.ceil(FLAGS.num_train_data / FLAGS.batch_size_per_GPU) + num_buckets).astype(np.int32)
    num_test_batches = (np.ceil(FLAGS.num_test_data / FLAGS.batch_size_per_GPU) + num_buckets).astype(np.int32)

    # train dataset
    ds_train = _bucket_and_batch(ds_train,
                                 bucket_boundaries)  # convert ds into batches of simmilar length features (bucketed)
    ds_train = ds_train.shuffle(buffer_size=FLAGS.buffer_size,
                                reshuffle_each_iteration=True)
    # TODO: perform DataAugmentation
    #  AdditiveNoise?
    #  TimeMasking
    #  FrequencyMasking
    ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

    # test dataset
    ds_test = _bucket_and_batch(ds_test, bucket_boundaries)
    ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

    return ds_train, ds_test, num_train_batches, num_test_batches


if __name__ == '__main__':
    ds_train, ds_test, num_train_batches, num_test_batches = load_datasets(FLAGS.load_dir)

    if ds_train:
        print(ds_train.output_shapes)

    if ds_test:
        print(ds_test.output_shapes)
