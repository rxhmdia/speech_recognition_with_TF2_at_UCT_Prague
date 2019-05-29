import os
import json
from datetime import datetime

import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import Reshape, Conv2D, Dropout, BatchNormalization, GRU, Bidirectional, Dense, ReLU, Permute, Lambda
from tensorflow.keras import Model
import tensorflow.keras.backend as K

from tqdm import tqdm

from DataPipeline import load_datasets
from FLAGS import FLAGS, PREDICTION_FLAGS
from utils import create_save_path, decay_value


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


def conv(x, n_channels, kernel_size, strides=(1, 1), dilation_rate=(1, 1), kernel_initializer=None, batch_norm=True, drop_rate=0.):
    x = Conv2D(filters=n_channels,
               kernel_size=kernel_size,
               strides=strides,
               padding=FLAGS.conv_params['padding'],
               data_format=FLAGS.conv_params['data_format'],
               dilation_rate=dilation_rate,
               kernel_initializer=kernel_initializer)(x)
    x = ReLU(max_value=FLAGS.relu_clip_val,
             negative_slope=FLAGS.relu_alpha)(x)
    if batch_norm:
        x = BatchNormalization(momentum=FLAGS.bn_momentum)(x)
    if 0. < drop_rate < 1.:
        x = Dropout(drop_rate)(x)
    return x


def rnn(x, num_units, kernel_initializer=None, batch_norm=True, drop_rate=0.):
    x = Bidirectional(GRU(num_units,
                          kernel_initializer=kernel_initializer,
                          recurrent_initializer=kernel_initializer,
                          return_sequences=True))(x)
    if batch_norm:
        x = BatchNormalization(momentum=FLAGS.bn_momentum)(x)
    if 0. < drop_rate < 1.:
        x = Dropout(drop_rate)(x)
    return x


def ff(x, num_units, kernel_initializer=None, batch_norm=True, drop_rate=0.):
    x = Dense(num_units, kernel_initializer=kernel_initializer)(x)
    x = ReLU(max_value=FLAGS.relu_clip_val,
             negative_slope=FLAGS.relu_alpha)(x)
    if batch_norm:
        x = BatchNormalization(momentum=FLAGS.bn_momentum)(x)
    if 0. < drop_rate < 1.:
        x = Dropout(drop_rate)(x)
    return x


def build_model(kernel_initializer):
    # Input 1
    x_in = tf.keras.Input(shape=(None, FLAGS.num_features))
    x = x_in
    # Feedforward layers at start
    if FLAGS.ff_first_params['use']:
        for ff_num_units, drop_rate in zip(FLAGS.ff_first_params['num_units'], FLAGS.ff_first_params['drop_rates']):
            x = ff(x, ff_num_units, kernel_initializer, FLAGS.ff_first_params['batch_norm'], drop_rate)
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
            x = conv(x, n_channels, kernel_size, strides, dilation_rate, kernel_initializer, FLAGS.conv_params['batch_norm'], drop_rate)
        x = Reshape((-1, int(tf.math.ceil(num_features)) * FLAGS.conv_params['channels'][-1]))(x)

    # Recurrent layers
    if FLAGS.rnn_params['use']:
        for rnn_num_units, drop_rate in zip(FLAGS.rnn_params['num_units'], FLAGS.rnn_params['drop_rates']):
            x = rnn(x, rnn_num_units, kernel_initializer, FLAGS.rnn_params['batch_norm'], drop_rate)

    # Feedforward layers at end
    if FLAGS.ff_params['use']:
        for ff_num_units, drop_rate in zip(FLAGS.ff_params['num_units'], FLAGS.ff_params['drop_rates']):
            x = ff(x, ff_num_units, kernel_initializer, FLAGS.ff_params['batch_norm'], drop_rate)

    # Output logits
    logits = Dense(FLAGS.alphabet_size + 1)(x)

    return Model(inputs=x_in, outputs=logits)


def early_stopping(model, cer, best_cer, epoch, best_epoch, save_path):
    """ Implementation of early stopping, which keeps the model with best CER throughout epochs and stops training
    if the cer doesn't improve on best_cer for FLAGS.patience_epochs

    :param model: model to save (or not)
    :param cer: (float) cer value of the current model
    :param best_cer: (float) cer value of the so far best model
    :param epoch: (int) current epoch
    :param best_epoch: (int) number of the currently best epoch (regarding best_cer)
    :param save_path: where to save the model
    :return: stop_training, best_cer, best_epoch
    """

    stop_training = False

    if cer < best_cer:
        model.save(os.path.join(save_path, 'model.h5'))
        best_cer = cer
        best_epoch = epoch
    else:
        if (epoch - best_epoch) >= FLAGS.patience_epochs:
            stop_training = True
    return stop_training, best_cer, best_epoch


@tf.function
def mean_ctc_loss(labels, logits, label_length, logit_length, name='ctc_loss'):
    loss = tf.nn.ctc_loss(labels, logits, label_length, logit_length,
                          logits_time_major=False,
                          blank_index=-1,
                          name=name)
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


def train_step(model, inputs, optimizer, time_reduce_rate):
    x, y, size_x, size_y = inputs

    logit_length = tf.cast(tf.math.ceil(time_reduce_rate*tf.cast(size_x, tf.float32)), tf.int32)
    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        mean_loss = mean_ctc_loss(y, logits, size_y, logit_length)
    gradients = tape.gradient(mean_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return mean_loss


def test_step(model, inputs, time_reduce_rate):
    x, y, size_x, size_y = inputs

    logit_length = tf.cast(tf.math.ceil(time_reduce_rate*tf.cast(size_x, tf.float32)), tf.int32)
    logits = model(x, training=False)
    mean_loss = mean_ctc_loss(y, logits, size_y, logit_length)
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
    """
    batch_no = 0
    # workaround for multiple runs of tf.function decorated function
    tf_train_step = tf.function(train_step)

    # decaying learning rate
    if FLAGS.lr_decay:
        new_lr = decay_value(FLAGS.lr, FLAGS.lr_decay_rate, FLAGS.lr_decay_epochs, epoch)
        print("____| Learning Rate {:.5f} |____".format(new_lr))
        K.set_value(optimizer.lr, new_lr)

    _, _, time_reduce_rate, _ = _conv_reduce_rate(FLAGS.max_time, FLAGS.num_features)
    train_loss = tf.keras.metrics.Mean(name='train_loss', dtype=tf.float32)
    with tqdm(range(num_batches), unit="batch") as timer:
        for inputs in dataset:
            mean_loss = tf_train_step(model, inputs, optimizer, time_reduce_rate)
            train_loss.update_state(mean_loss)
            timer.update(1)
            if tf.equal(batch_no % 100, 0):
                print("Batch {} | Loss {}".format(batch_no, mean_loss))
            batch_no += 1
    mean_train_loss = train_loss.result()
    print("Mean Loss: {}".format(mean_train_loss))
    tf.summary.scalar('mean_loss', mean_train_loss, step=epoch, description='mean train loss')
    train_loss.reset_states()

    return mean_train_loss


def test_fn(model, dataset, num_batches, epoch):
    batch_no = 0
    _, _, time_reduce_rate, _ = _conv_reduce_rate(FLAGS.max_time, FLAGS.num_features)
    test_loss = tf.keras.metrics.Mean(name='test_loss', dtype=tf.float32)
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
    mean_test_loss = test_loss.result()
    mean_test_cer = test_cer.result()
    print("Mean Loss: {}".format(mean_test_loss))
    print("Mean CER: {}".format(mean_test_cer))
    tf.summary.scalar('mean_loss', mean_test_loss, step=epoch, description='mean test loss')
    tf.summary.scalar('mean_cer', mean_test_cer, step=epoch, description='mean test character error rate')
    test_loss.reset_states()
    test_cer.reset_states()

    return mean_test_loss, mean_test_cer


def train_model(run_number):
    K.clear_session()
    save_path = create_save_path()

    # __INPUT PIPELINE__ #
    ds_train, ds_test, num_train_batches, num_test_batches = load_datasets(FLAGS.load_dir)

    # __MODEL__ #
    kernel_initializer = tf.initializers.TruncatedNormal(mean=FLAGS.weight_init_mean,
                                                         stddev=FLAGS.weight_init_stddev)
    model = build_model(kernel_initializer)
    #        print('Trainable params: {}'.format(model.count_params()))
    print(model.summary())

    # Load model weights from checkpoint if checkpoint_path is provided
    if FLAGS.checkpoint_path:
        model.load_weights(FLAGS.checkpoint_path)

    # save model config (FLAGS) to JSON file
    with open(save_path + '/config.json', 'w') as f:
        json.dump(
            {key: value for key, value in FLAGS.__dict__.items() if not key.startswith('__') and not callable(key)}, f)

    # save model architecture image to save_path directory
    if FLAGS.save_architecture_image:
        tf.keras.utils.plot_model(model, os.path.join(save_path, 'architecture.png'), show_shapes=FLAGS.show_shapes)

    # __LOGGING__ #
    summary_writer_train = tf.summary.create_file_writer(save_path + '/train', name='sw-train')
    summary_writer_test = tf.summary.create_file_writer(save_path + '/test', name='sw-test')

    # __TRAINING__ #
    best_cer = 100.0
    best_epoch = -1
    optimizer = tf.keras.optimizers.Adam(lr=FLAGS.lr, epsilon=FLAGS.epsilon, amsgrad=FLAGS.amsgrad)

    for epoch in range(FLAGS.max_epochs):
        print('_______| Run {} | Epoch {} |_______'.format(run_number, epoch))

        # TRAINING DATA
        with summary_writer_train.as_default():
            train_fn(model, ds_train, optimizer, num_train_batches, save_path, epoch)

        # TESTING DATA
        with summary_writer_test.as_default():
            _, test_cer = test_fn(model, ds_test, num_test_batches, epoch)

        # EARLY STOPPING AND KEEPING THE BEST MODEL
        stop_training, best_cer, best_epoch = early_stopping(model, test_cer, best_cer, epoch, best_epoch, save_path)
        print('| Best CER {} | Best epoch {} |'.format(best_cer, best_epoch))
        if stop_training:
            print('Model stopped early at epoch {}'.format(epoch))
            break


def predict_from_saved_model(path_to_model, feature_inputs, beam_width=PREDICTION_FLAGS.beam_width,
                             top_paths=PREDICTION_FLAGS.top_paths):
    """ Load model from path_to_model, calculate logits from feature_input and decode outputs
    to produce prediction string

    :param path_to_model: (str) path to .h5 saved model
    :param feature_inputs: (numpy float array or list of numpy float arrays)
    :param beam_width: (int) beam width of ctc beam search decoder
    :param top_paths: (int) number of best paths to be predicted

    :return predictions: (List[str]) string transcriptions of the predictions
    """

    predictions = []

    model = tf.keras.models.load_model(path_to_model, custom_objects={'tf': tf}, compile=False)

    if isinstance(feature_inputs, np.ndarray):
        inputs = [feature_inputs]
    elif isinstance(feature_inputs, list):
        inputs = []
        for i, x in enumerate(inputs):
            if isinstance(x, np.ndarray):
                inputs.append(x)
            else:
                print('ignoring input number {} as it is not a numpy array'.format(i))
    else:
        raise TypeError('feature_inputs argument is not a numpy array or a list of numpy arrays')

    for i, x in enumerate(inputs):
        logits = model(tf.expand_dims(tf.cast(x, tf.float32), 0), training=False)
        logits_time_major = tf.transpose(logits, (1, 0, 2))
        decoded, _ = tf.nn.ctc_beam_search_decoder(logits_time_major,
                                                   [tf.shape(logits_time_major)[0]],
                                                   beam_width=beam_width,
                                                   top_paths=top_paths)

        dense_decoded = [tf.sparse.to_dense(d, default_value=-1) for d in decoded]

        for j, decoded_path in enumerate(dense_decoded):
            print("Prediction {} | Path {}: {}".format(i, j, "".join([PREDICTION_FLAGS.n2c_map[int(c)]
                                                                      for c in decoded_path[0, :] if int(c) != -1])))

        predictions.append(dense_decoded)

    return predictions

if __name__ == '__main__':
    K.clear_session()

    kernel_initializer = tf.initializers.TruncatedNormal(mean=FLAGS.weight_init_mean,
                                                         stddev=FLAGS.weight_init_stddev)

    model = build_model(kernel_initializer)

    model.summary()
    tf.keras.utils.plot_model(model, 'AcousticModel.png', show_shapes=True)





