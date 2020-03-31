import os
import json
import shutil
from datetime import datetime

from FLAGS import FLAGS


def _layer_param_format(name, condition, param_list, batch_norm):
    if condition:
        return '_{}[{}]{}'.format(name, '-'.join(str(p) for p in param_list), '(bn)' if batch_norm else '')
    else:
        return ''


def create_save_path(exist_ok=True):
    time_id = datetime.now().strftime('%Y-%m%d-%H%M%S')  # unique episode id from current date and time
    ep_name = ''.join([time_id,
                       _layer_param_format('ff_first', FLAGS.ff_first_params['use'], FLAGS.ff_first_params['num_units'],
                                           FLAGS.ff_first_params['batch_norm']),
                       _layer_param_format('conv', FLAGS.conv_params['use'], FLAGS.conv_params['channels'],
                                           FLAGS.conv_params['batch_norm']),
                       _layer_param_format('rnn', FLAGS.rnn_params['use'], FLAGS.rnn_params['num_units'],
                                           FLAGS.rnn_params['batch_norm']),
                       _layer_param_format('ff', FLAGS.ff_params['use'], FLAGS.ff_params['num_units'],
                                           FLAGS.ff_params['batch_norm']),
                       ])

    save_path = os.path.join(FLAGS.save_dir, ep_name)

    os.makedirs(save_path, exist_ok=exist_ok)

    return save_path


def save_config(save_path):
    name, ext = os.path.splitext(FLAGS.save_config_as)
    path_to_file = '{}/{}{}'.format(save_path, name, ext)
    if ext == '.json':
        with open(path_to_file, 'w') as f:
            json.dump(
                {key: value for key, value in FLAGS.__dict__.items()
                 if not key.startswith('__') and not callable(key)}, f)
    elif ext == '.py':
        shutil.copy2('./FLAGS.py', path_to_file)
    else:
        raise FileNotFoundError("Extention of 'FLAGS.save_config_as' must be either .json or .py")


def decay_value(base_value, decay_rate, decay_steps, step):
    """ decay base_value by decay_rate every decay_steps

    :param base_value:
    :param decay_rate:
    :param decay_steps:
    :param step:
    :return: decayed value
    """
    return base_value*decay_rate**(step/decay_steps)
