import os
from datetime import datetime

from FLAGS import FLAGS


def create_save_path(exist_ok=True):
    # TODO: move to helpers
    time_id = datetime.now().strftime('%Y-%m%d-%H%M%S')  # unique episode id from current date and time
    ep_name = '{}_conv[{}]{}_rnn[{}]{}_ff[{}]{}'.format(time_id,
                                                        '-'.join(str(p) for p in FLAGS.conv_params['channels']),
                                                        '(bn)' if FLAGS.conv_params['batch_norm'] else '',
                                                        '-'.join(str(p) for p in FLAGS.rnn_params['num_units']),
                                                        '(bn)' if FLAGS.rnn_params['batch_norm'] else '',
                                                        '-'.join(str(p) for p in FLAGS.ff_params['num_units']),
                                                        '(bn)' if FLAGS.ff_params['batch_norm'] else '')

    save_path = os.path.join(FLAGS.save_dir, ep_name)

    os.makedirs(save_path, exist_ok=exist_ok)

    return save_path
