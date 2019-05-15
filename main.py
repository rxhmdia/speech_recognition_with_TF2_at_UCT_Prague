from FLAGS import FLAGS
from Model import train_model

if __name__ == '__main__':
    for run_number in range(FLAGS.num_runs):
        train_model(run_number)
