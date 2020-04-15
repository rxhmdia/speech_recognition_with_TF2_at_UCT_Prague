import tensorflow as tf

from DataOps import load_datasets
from FLAGS import FLAGS

if __name__ == '__main__':
    from matplotlib import pyplot as plt
    ds_train, ds_test, num_train_batches, num_test_batches = load_datasets(FLAGS.load_dir)

    epochs = 2

    if ds_train:
        for epoch in range(epochs):
            for i, sample in enumerate(ds_train):
                if i % 500 == 0:
                    plt.figure()
                    plt.pcolormesh(tf.transpose(sample[0][0, :, :], (1, 0)))
                    print(sample[0].shape)
            print(ds_train)

    plt.show()

    if ds_test:
        print(ds_test)