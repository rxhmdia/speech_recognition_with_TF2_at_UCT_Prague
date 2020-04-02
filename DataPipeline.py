import os

import numpy as np
import tensorflow as tf

from helpers import console_logger
from FLAGS import FLAGS

LOGGER = console_logger('tensorflow', FLAGS.logger_level)
_AUTOTUNE = tf.data.experimental.AUTOTUNE


# TODO:
#  AdditiveNoise?
# DONE:
#  SpecAug doesn't work with None shapes (so for time masking, the current code doesn't work)
#  Batching only works if the explicit shapes in batch are all the same (freq mask works only without random bandwidth)
#  Ensure resulting frequency shape is the same before and after masking? (pad with zeros instead of removing)
#  TimeMasking
#  FrequencyMasking
#  multiple instances of SpecAug (2x TimeMasking, 2x TimeMasking)

# noinspection DuplicatedCode
class SpecAug:

    def __init__(self, axis=0, bandwidth=20):
        """ Tensorflow data pipeline implementation of SpecAug time and frequency masking

        :param axis (int): which axis will be masked (0 ... time, 1 ... frequency)
        :param bandwidth (int): length of the masked area
        """
        self.axis = axis if axis in (0, 1) else 0
        self.bandwidth = bandwidth
        self._max_sx = None

    @tf.function(experimental_relax_shapes=True)
    def _mask_sample(self, sample, sx_max=None):
        x, y, sx, sy = sample
        stime, sfreq = (sx, x.shape[1])

        if self.axis == 0:
            full_len = stime
        elif self.axis == 1:
            full_len = sfreq
        else:
            raise AttributeError("self.axis must be either 0 (time masking) or 1 (frequency masking)")

        # generate position of masking
        bandwidth = self.bandwidth
        tm_lb = tf.random.uniform([], 0, full_len - bandwidth, dtype=tf.int32)  # lower bounds
        tm_ub = tm_lb + bandwidth  # upper bounds

        # generate lower bound and upper bound masks
        mask_lb = tf.concat((tf.ones([tm_lb, ], dtype=tf.bool), tf.zeros([full_len - tm_lb, ], dtype=tf.bool)), axis=0)
        mask_ub = tf.concat((tf.zeros([tm_ub, ], dtype=tf.bool), tf.ones([full_len - tm_ub], dtype=tf.bool)), axis=0)

        # get value for padding batch to same time length
        padding = sx_max - stime

        if self.axis == 0:
            # TIME MASKING
            x = tf.concat((tf.boolean_mask(x, mask_lb, axis=0),
                           tf.zeros([bandwidth + padding, sfreq]),
                           tf.boolean_mask(x, mask_ub, axis=0)), axis=0)
        elif self.axis == 1:
            # FREQUENCY MASKING
            # x = tf.pad(x, [[0, padding], [0, 0]])
            x = tf.concat((tf.boolean_mask(x, mask_lb, axis=1),
                           tf.zeros([sx_max, bandwidth]),
                           tf.boolean_mask(x, mask_ub, axis=1)), axis=1)
        else:
            raise AttributeError("self.axis must be either 0 (time masking) or 1 (frequency masking)")

        # sx = sx + padding
        x = tf.ensure_shape(x, (None, sfreq))

        return x, y, sx, sy

    @tf.function(experimental_relax_shapes=True)
    def mask(self, x, y, sx, sy):
        return tf.map_fn(lambda sample: self._mask_sample(sample, tf.reduce_max(sx)),
                         (x, y, sx, sy),
                         parallel_iterations=4)


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


# noinspection PyShadowingNames
def load_datasets(load_dir,
                  data_aug=FLAGS.data_aug['mode'],
                  bandwidth_time=FLAGS.data_aug['bandwidth_time'],
                  bandwidth_freq=FLAGS.data_aug['bandwidth_freq']):
    path_gen = os.walk(load_dir)

    ds_train = None
    ds_test = None

    if '1x' in data_aug or '2x' in data_aug:
        LOGGER.info("Initializing Data Augmentation for time and freq.")
        sa_time = SpecAug(axis=0, bandwidth=bandwidth_time)
        sa_freq = SpecAug(axis=1, bandwidth=bandwidth_freq)
        LOGGER.debug(f"sa_time.bandwidth: {sa_time.bandwidth} |"
                     f"sa_freq.bandwidth: {sa_freq.bandwidth}")
    else:
        LOGGER.info("Skipping Pipeline Data Augmentation.")
        sa_time = None
        sa_freq = None

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
            LOGGER.info(f"joined dataset loaded from {path} and split into ds_train ({FLAGS.num_train_data}) "
                        f"and ds_test (rest)")
            break
        if folder_name == 'test':
            ds_test = _read_tfrecords(fullpaths, block_length=FLAGS.num_test_data)
            LOGGER.info(f'test dataset loaded from {path}')
        elif folder_name == 'train':
            # don't shuffle if using shards, because bucketting doesn't work well with shards
            ds_train = _read_tfrecords(fullpaths, block_length=FLAGS.num_train_data)
            LOGGER.info(f'train dataset loaded from {path}')
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
    # DATA AUGMENTATION
    if '2x' in data_aug:
        ds_train = (ds_train.map(sa_time.mask, num_parallel_calls=_AUTOTUNE)   # time masking 1
                            .map(sa_time.mask, num_parallel_calls=_AUTOTUNE)   # time masking 2
                            .map(sa_freq.mask, num_parallel_calls=_AUTOTUNE)   # frequency masking 1
                            .map(sa_freq.mask, num_parallel_calls=_AUTOTUNE))  # frequency masking 2
    elif '1x' in data_aug:
        ds_train = (ds_train.map(sa_time.mask, num_parallel_calls=_AUTOTUNE)  # time masking
                            .map(sa_freq.mask, num_parallel_calls=_AUTOTUNE))  # frequency masking
    else:
        LOGGER.info("Data Augmentation NOT added into pipeline.")
    ds_train = ds_train.shuffle(buffer_size=FLAGS.buffer_size,
                                reshuffle_each_iteration=True)
    ds_train = ds_train.prefetch(_AUTOTUNE)

    # test dataset
    ds_test = _bucket_and_batch(ds_test, bucket_boundaries)
    ds_test = ds_test.prefetch(_AUTOTUNE)

    return ds_train, ds_test, num_train_batches, num_test_batches


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    ds_train, ds_test, num_train_batches, num_test_batches = load_datasets(FLAGS.load_dir)

    epochs = 2

    if ds_train:
        for epoch in range(epochs):
            for i, sample in enumerate(ds_train):
                print(sample[0].shape)
                if i % 500 == 0:
                    plt.figure()
                    plt.pcolormesh(tf.transpose(sample[0][0, :, :], (1, 0)))
            print(ds_train)

    plt.show()

    if ds_test:
        print(ds_test)
