import os
import json
from datetime import datetime

import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import Reshape, Conv2D, Dropout, BatchNormalization, GRU, Bidirectional, Dense, ReLU, Permute, Lambda
from tensorflow.keras import Model
import tensorflow.keras.backend as K

from tqdm import tqdm

from FLAGS import FLAGS


def _conv_output_shape(input_shape, filt_shape, filt_stride, padding="same"):
    if padding.upper() == "SAME":
        time_size, feat_size = (1, 1)
    elif padding.upper() == "VALID":
        time_size, feat_size = filt_shape
    else:
        raise ValueError('padding argument must be SAME or VALID')
    output_size = lambda size, filt_size, filt_stride: tf.math.divide(
        tf.math.divide(size - filt_size + 1, filt_stride), 1)
    time_output_size = output_size(input_shape[0], time_size, filt_stride[0])
    feat_output_size = output_size(input_shape[1], feat_size, filt_stride[1])
    return time_output_size, feat_output_size


def _conv_reduce_rate(max_time, num_features):
    for kernel_size, strides in zip(FLAGS.conv_params['kernels'], FLAGS.conv_params['strides']):
        max_time, num_features = _conv_output_shape((max_time, num_features), kernel_size, strides,
                                                    FLAGS.conv_params['padding'])

    return max_time, num_features, max_time/FLAGS.max_time, num_features/FLAGS.num_features


def conv(x, n_channels, kernel_size, strides=(1, 1), dilation_rate=(1, 1), batch_norm=True, drop_rate=0.):
    x = Conv2D(filters=n_channels,
               kernel_size=kernel_size,
               strides=strides,
               padding=FLAGS.conv_params['padding'],
               data_format=FLAGS.conv_params['data_format'],
               dilation_rate=dilation_rate)(x)
    x = ReLU(max_value=FLAGS.relu_clip_val,
             negative_slope=FLAGS.relu_alpha)(x)
    if batch_norm:
        x = BatchNormalization(momentum=FLAGS.bn_momentum)(x)
    if 0. < drop_rate < 1.:
        x = Dropout(drop_rate)(x)
    return x


def rnn(x, num_units, batch_norm=True, drop_rate=0.):
    x = Bidirectional(GRU(num_units, return_sequences=True))(x)
    if batch_norm:
        x = BatchNormalization(momentum=FLAGS.bn_momentum)(x)
    if 0. < drop_rate < 1.:
        x = Dropout(drop_rate)(x)
    return x


def ff(x, num_units, batch_norm=True, drop_rate=0.):
    x = Dense(num_units)(x)
    x = ReLU(max_value=FLAGS.relu_clip_val,
             negative_slope=FLAGS.relu_alpha)(x)
    if batch_norm:
        x = BatchNormalization(momentum=FLAGS.bn_momentum)(x)
    if 0. < drop_rate < 1.:
        x = Dropout(drop_rate)(x)
    return x


def build_model():
    # Input 1
    x_in = tf.keras.Input(shape=(None, FLAGS.num_features))
    x = x_in
    # Feedforward layers at start
    if FLAGS.ff_first_params['use']:
        for ff_num_units, drop_rate in zip(FLAGS.ff_first_params['num_units'], FLAGS.ff_first_params['drop_rates']):
            x = ff(x, ff_num_units, FLAGS.ff_first_params['batch_norm'], drop_rate)
        num_features = FLAGS.ff_first_params['num_units'][-1]
    else:
        num_features = FLAGS.num_features

    # Convolutional layers
    if FLAGS.conv_params['use']:
        x = Lambda(lambda x: tf.expand_dims(x, -1))(x)
        _, num_features, _, _ = _conv_reduce_rate(FLAGS.max_time, num_features)
        for n_channels, kernel_size, strides, dilation_rate, drop_rate in zip(FLAGS.conv_params['channels'],
                                                                              FLAGS.conv_params['kernels'],
                                                                              FLAGS.conv_params['strides'],
                                                                              FLAGS.conv_params['dilation_rates'],
                                                                              FLAGS.conv_params['drop_rates']):
            x = conv(x, n_channels, kernel_size, strides, dilation_rate, FLAGS.conv_params['batch_norm'], drop_rate)
        x = Reshape((-1, int(tf.math.ceil(num_features)) * FLAGS.conv_params['channels'][-1]))(x)

    # Recurrent layers
    if FLAGS.rnn_params['use']:
        for rnn_num_units, drop_rate in zip(FLAGS.rnn_params['num_units'], FLAGS.rnn_params['drop_rates']):
            x = rnn(x, rnn_num_units, FLAGS.rnn_params['batch_norm'], drop_rate)

    # Feedforward layers at end
    if FLAGS.ff_params['use']:
        for ff_num_units, drop_rate in zip(FLAGS.ff_params['num_units'], FLAGS.ff_params['drop_rates']):
            x = ff(x, ff_num_units, FLAGS.ff_params['batch_norm'], drop_rate)

    # Output logits
    logits = Dense(FLAGS.alphabet_size + 1)(x)

    return Model(inputs=x_in, outputs=logits)


@tf.function
def mean_ctc_loss(labels, logits, label_length, logit_length, implementation='keras', name='ctc_loss'):
    if implementation == 'keras':
        logit_len = tf.expand_dims(logit_length, -1)
        label_len = tf.expand_dims(label_length, -1)
        loss = K.ctc_batch_cost(labels, logits, logit_len, label_len)
    elif implementation == 'tensorflow':
        loss = tf.nn.ctc_loss(labels, logits, label_length, logit_length,
                              logits_time_major=False,
                              blank_index=-1,
                              name=name)
    else:
        raise ValueError("Unknown implementation (must be either 'keras' or 'tensorflow')")
    return tf.reduce_mean(loss / tf.cast(logit_length, tf.float32))  # time length normalized loss


def decode_best_output(labels, logits, label_length, logit_length):
    # transpose from batch-major to time-major
    logits_time_major = tf.transpose(logits, (1, 0, 2))
    decoded, log_probs = tf.nn.ctc_beam_search_decoder(logits_time_major,
                                                       logit_length,
                                                       beam_width=FLAGS.beam_width,
                                                       top_paths=FLAGS.top_paths)

    decoded = [tf.cast(d, tf.int32) for d in decoded]
    truth = K.ctc_label_dense_to_sparse(tf.cast(labels, tf.int32), label_length)

    cer = tf.edit_distance(decoded[0], truth, name="levenshtein_distance")
    cer = tf.boolean_mask(cer, tf.not_equal(cer, np.inf))  # remove inf values (where the length of labels == 0)

    mean_cer = tf.reduce_mean(cer)

    return tf.sparse.to_dense(decoded[0], default_value=FLAGS.label_pad_val), mean_cer


@tf.function
def train_step(model, inputs, optimizer, time_reduce_rate):
    x, y, size_x, size_y = inputs

    logit_length = tf.cast(tf.math.ceil(time_reduce_rate*tf.cast(size_x, tf.float32)), tf.int32)
    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        mean_loss = mean_ctc_loss(y, logits, size_y, logit_length,
                                  implementation='tensorflow')
    gradients = tape.gradient(mean_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return mean_loss


def test_step(model, inputs, time_reduce_rate):
    x, y, size_x, size_y = inputs

    logit_length = tf.cast(tf.math.ceil(time_reduce_rate*tf.cast(size_x, tf.float32)), tf.int32)
    logits = model(x, training=False)
    mean_loss = mean_ctc_loss(y, logits, size_y, logit_length,
                              implementation='tensorflow')
    decoded, mean_cer = decode_best_output(y, logits, size_y, logit_length)

    #    print('decoded: {}'.format(decoded))
    #    print('truth: {}'.format(y))
    #    print('mean_loss: {}'.format(mean_loss))
    #    print('mean_cer: {}'.format(mean_cer))

    return mean_loss, decoded, mean_cer


def train_fn(model, dataset, optimizer, num_batches, save_path, epoch):
    """
    One epoch of training:
     - train 'model' on 'dataset' using 'optimizer'
     - show loss every 100 batches
     - write total_loss to TB summary to 'save_path/train'
     - save the trained model to 'save_path' """
    batch_no = 0
    _, _, time_reduce_rate, _ = _conv_reduce_rate(FLAGS.max_time, FLAGS.num_features)
    train_loss = tf.keras.metrics.Sum(name='train_loss', dtype=tf.float32)
    with tqdm(range(num_batches), unit="batch") as timer:
        for inputs in dataset:
            mean_loss = train_step(model, inputs, optimizer, time_reduce_rate)
            train_loss.update_state(mean_loss)
            timer.update(1)
            if tf.equal(batch_no % 100, 0):
                print("Batch {} | Loss {}".format(batch_no, mean_loss))
            batch_no += 1
    print("Total Loss: {}".format(train_loss.result()))
    tf.summary.scalar('total_loss', train_loss.result(), step=epoch, description='total train loss')
    train_loss.reset_states()
    model.save(os.path.join(save_path, 'model-ep-{}.h5'.format(epoch)))


def test_fn(model, dataset, num_batches, epoch):
    batch_no = 0
    _, _, time_reduce_rate, _ = _conv_reduce_rate(FLAGS.max_time, FLAGS.num_features)
    test_loss = tf.keras.metrics.Sum(name='test_loss', dtype=tf.float32)
    test_cer = tf.keras.metrics.Mean(name='test_cer', dtype=tf.float32)
    with tqdm(range(num_batches), unit="batch") as timer:
        for inputs in dataset:
            mean_loss, decoded, mean_cer = test_step(model, inputs, time_reduce_rate)
            test_loss.update_state(mean_loss)
            test_cer.update_state(mean_cer)
            timer.update(1)
            if tf.equal(batch_no % 100, 0):
                print("\nBatch {} | Loss {} | CER {}".format(batch_no, mean_loss, mean_cer))
                print("Prediction: {}".format("".join([FLAGS.n2c_map[int(c)] for c in decoded[0, :] if int(c) != -1])))
                print("Truth: {}".format("".join([FLAGS.n2c_map[int(c)] for c in inputs[1][0, :] if int(c) != -1])))
            batch_no += 1
    print("Total Loss: {}".format(test_loss.result()))
    print("Mean CER: {}".format(test_cer.result()))
    tf.summary.scalar('total_loss', test_loss.result(), step=epoch, description='total test loss')
    tf.summary.scalar('mean_cer', test_cer.result(), step=epoch, description='mean test character error rate')
    test_loss.reset_states()
    test_cer.reset_states()


if __name__ == '__main__':
    K.clear_session()

    model = build_model()

    model.summary()
    tf.keras.utils.plot_model(model, 'AcousticModel.png', show_shapes=True)





