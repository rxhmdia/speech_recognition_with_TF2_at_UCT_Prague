import json
import os

import tensorflow as tf

from FLAGS import FLAGS
from DataPipeline import load_datasets
from Model import build_model, train_fn, test_fn
from utils import create_save_path

if __name__ == '__main__':
    for run_number in range(FLAGS.num_runs):
        tf.keras.backend.clear_session()
        ds_train, ds_test, num_train_batches, num_test_batches = load_datasets(FLAGS.load_dir)
        model = build_model()
#        print('Trainable params: {}'.format(model.count_params()))
        print(model.summary())

        # Load model weights from checkpoint if checkpoint_path is provided
        if FLAGS.checkpoint_path:
            model.load_weights(FLAGS.checkpoint_path)

        save_path = create_save_path()

        summary_writer_train = tf.summary.create_file_writer(save_path + '/train', name='sw-train')
        summary_writer_test = tf.summary.create_file_writer(save_path + '/test', name='sw-test')

        # save model config to json file
        with open(save_path + '/config.json', 'w') as f:
            json.dump(
                {key: value for key, value in FLAGS.__dict__.items() if not key.startswith('__') and not callable(key)}, f)

        # save model architecture image to save_dir
        if FLAGS.save_architecture_image:
            tf.keras.utils.plot_model(model, os.path.join(save_path, 'architecture.png'), show_shapes=FLAGS.show_shapes)

        optimizer = tf.keras.optimizers.Adam(lr=FLAGS.lr, epsilon=FLAGS.epsilon, amsgrad=FLAGS.amsgrad)

        for epoch in range(FLAGS.max_epochs):
            print('_______| Run {} | Epoch {} |_______'.format(run_number, epoch))
            # TRAINING DATA
            with summary_writer_train.as_default():
                train_fn(model, ds_train, optimizer, num_train_batches, save_path, epoch)
            # TESTING DATA
            with summary_writer_test.as_default():
                test_fn(model, ds_test, num_test_batches, epoch)
