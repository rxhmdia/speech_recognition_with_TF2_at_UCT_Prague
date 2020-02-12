import os

from FLAGS import FLAGS
from Model import train_model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

if __name__ == '__main__':
    for run_number in range(FLAGS.num_runs):
        train_model(run_number, logger_level="DEBUG")
