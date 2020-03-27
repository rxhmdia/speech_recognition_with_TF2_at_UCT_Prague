import time

import numpy as np
import tensorflow as tf
from helpers import console_logger

_NUM_SAMPLES = 20
_DATA_SHAPE = (10, 5)  # time, num_features
_BATCH_SIZE = 1
_NUM_INSTANCES = 1  # > 1 not supported
_BANDWIDTH = 2

LOGGER = console_logger(__name__, "DEBUG")

# TODO:
#  - TimeMasking
#  - FrequencyMasking


class ArtificialDataset(tf.data.Dataset):
    def _generator(num_samples):
        # Opening the file
        time.sleep(0.03)

        for sample_idx in range(num_samples):
            # Reading data (line, record) from the file
            time.sleep(0.015)

            yield np.random.standard_normal(_DATA_SHAPE)

    def __new__(cls, num_samples=_NUM_SAMPLES):
        return tf.data.Dataset.from_generator(
            cls._generator,
            output_types=tf.dtypes.float64,
            output_shapes=_DATA_SHAPE,
            args=(num_samples,)
        )


# TODO: make into classes
#  _make_sample will be mother class
#  _timeaug will be one subclass
#  _freqaug will be second sublass


def _mask_sample(sample, num_instances=_NUM_INSTANCES, bandwidth=_BANDWIDTH):
    # tm_lb = np.random.randint(0, sample.shape[0]-bandwidth)
    # tm_ub = tm_lb + bandwidth

    for i in range(num_instances):
        ntime, nfreq = sample.shape
        tm_lb = tf.random.uniform([], 0, ntime-bandwidth, dtype=tf.int32)  # lower bounds
        tm_ub = tm_lb + bandwidth  # upper bounds
        # LOGGER.info(f"tm_lb: {tm_lb}, tm_ub: {tm_ub}")

        mask = tf.concat((tf.ones((tm_lb, ), dtype=tf.bool),
                         tf.zeros((bandwidth, ), dtype=tf.bool),
                         tf.ones((ntime-tm_ub, ), dtype=tf.bool)), axis=0)
        # LOGGER.debug(f"tm.shape: {mask.shape}")

        sample = tf.boolean_mask(sample, mask)

    return sample


@tf.function
def mask_time(x):
    return tf.map_fn(_mask_sample, x, parallel_iterations=4)


def freqmasking():
    pass


if __name__ == '__main__':
    ds = ArtificialDataset().batch(_BATCH_SIZE).map(mask_time, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    for sample in ds:
        LOGGER.debug(f"sample.shape: {sample.shape}")
    LOGGER.info("終わりました")
