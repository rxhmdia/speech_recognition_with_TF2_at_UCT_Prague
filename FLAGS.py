import json
from helpers import console_logger


class FLAGS:
    # noinspection DuplicatedCode
    use_digits = False
    if use_digits:
        c2n_map = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
                   'a': 10, 'á': 11, 'b': 12, 'c': 13, 'č': 14, 'd': 15, 'ď': 16, 'e': 17, 'é': 18, 'ě': 19,
                   'f': 20, 'g': 21, 'h': 22, 'ch': 23, 'i': 24, 'í': 25, 'j': 26, 'k': 27, 'l': 28, 'm': 29,
                   'n': 30, 'ň': 31, 'o': 32, 'ó': 33, 'p': 34, 'q': 35, 'r': 36, 'ř': 37, 's': 38, 'š': 39,
                   't': 40, 'ť': 41, 'u': 42, 'ú': 43, 'ů': 44, 'v': 45, 'w': 46, 'x': 47, 'y': 48, 'ý': 49,
                   'z': 50, 'ž': 51, ' ': 52}
    else:
        c2n_map = {'a': 0, 'á': 1, 'b': 2, 'c': 3, 'č': 4, 'd': 5, 'ď': 6, 'e': 7, 'é': 8, 'ě': 9,
                   'f': 10, 'g': 11, 'h': 12, 'ch': 13, 'i': 14, 'í': 15, 'j': 16, 'k': 17, 'l': 18, 'm': 19,
                   'n': 20, 'ň': 21, 'o': 22, 'ó': 23, 'p': 24, 'q': 25, 'r': 26, 'ř': 27, 's': 28, 'š': 29,
                   't': 30, 'ť': 31, 'u': 32, 'ú': 33, 'ů': 34, 'v': 35, 'w': 36, 'x': 37, 'y': 38, 'ý': 39,
                   'z': 40, 'ž': 41, ' ': 42}
    n2c_map = {val: idx for idx, val in c2n_map.items()}
    alphabet_size = len(c2n_map)

    # Character-level map for encoder-decoder language models (0 is padding for this one!!!!)
    c2n_map_lm = {'<pad>': 0}
    for k, v in c2n_map.items():
        c2n_map_lm[k] = v + 1
    c2n_map_lm["<sos>"] = len(c2n_map_lm)
    c2n_map_lm["<eos>"] = len(c2n_map_lm)
    n2c_map_lm = {val: key for key, val in c2n_map_lm.items()}

    logger_level = "INFO"
    logger = console_logger(__name__, logger_level)

    load_dir = "b:/!temp/PDTSC_MFSC_unigram_40_banks_DEBUG_min_100_max_3000_tfrecord/1.0/"
    # load_dir = "g:/datasets/PDTSC_Debug/"
    # load_dir = "g:/datasets/PDTSC_MFSC_unigram_40_banks_min_100_max_3000_tfrecord/"
    # load_dir = "g:/datasets/PDTSC_MFSC_unigram_40_banks_min_100_max_3000_tfrecord_DAspeed/"
    # load_dir = "g:/datasets/ORAL_MFSC_unigram_40_banks_min_100_max_3000_tfrecord/"
    # load_dir = "g:/datasets/COMBINED_MFSC_unigram_40_banks_min_100_max_3000_tfrecord/1.0/"
    save_dir = "./results/"
    save_config_as = "FLAGS.py"
    checkpoint_path = None
    # Tried to remove LM, see what it does now
    # Removing LM didn't help so its somewhere else. Removed softmax from final activations.
    #  Next - Trying to add LM without BN and DP
    #  Changed lr from 0.01 to 0.1
    # TODO:
    #  Add BN and DP to LM
    num_runs = 2
    max_epochs = 60
    batch_size_per_GPU = 8

    fs = 16000  # sampling rate of the loaded audiofiles

    # noinspection DuplicatedCode
    try:
        with open(load_dir + "data_config.json", "r") as f:
            dc = json.load(f)
            num_train_data = dc["num_train_data"]  # int(48812/2)  # int(11308/2)  # full ORAL == 374714/2
            num_test_data = dc["num_test_data"]  # int(4796/2)  # int(1304/2)  # full ORAL == 16502/2
            num_features = dc["num_features"]  # 123
            min_time = dc["min_time"]  # 100
            max_time = dc["max_time"]  # 3000
    except FileNotFoundError:
        logger.warning(f"data_config.json file not found at {load_dir}. Loading default values.")
        num_train_data = 24406
        num_test_data = 2398
        num_features = 123
        min_time = 100
        max_time = 3000

    buffer_size = int(0.1*num_train_data/batch_size_per_GPU)
    shuffle_seed = 42

    bucket_width = 100

    feature_pad_val = 0.0
    label_pad_val = -1
    label_pad_val_lm = 0

    # MODEL
    save_architecture_image = False
    show_shapes = True

    weight_init_mean = 0.0
    weight_init_stddev = 0.0001

    ff_first_params = {
        'use': False,
        'num_units': [128, 64],
        'batch_norm': False,
        'drop_rates': [0.],
    }
    conv_params = {
        'use': True,
        'channels': [32, 64, 128, 256],
        'kernels': [(16, 32), (8, 16), (4, 8), (4, 4)],
        'strides': [(2, 4), (2, 4), (1, 2), (1, 2)],
        'dilation_rates': [(1, 1), (1, 1), (1, 1), (1, 1)],
        'padding': 'same',
        'data_format': 'channels_last',
        'batch_norm': True,
        'drop_rates': [0., 0., 0., 0.],
    }
    bn_momentum = 0.9
    relu_clip_val = 20
    relu_alpha = 0.2
    rnn_params = {
        'use': True,
        'num_units': [512, 512],
        'batch_norm': True,
        'drop_rates': [0., 0.],
    }
    ff_params = {
        'use': True,
        'num_units': [256, 128],
        'batch_norm': True,
        'drop_rates': [0., 0.],
    }

    # LANGUAGE MODEL params
    # simple language model
    lm_gru_params = {
        'use': True,
        'num_units': [128, 64],
        'batch_norm': False,
        'drop_rates': [0.0, 0.0]
    }

    # encoder-decoder model
    enc_dec_hyperparams = {
        'train_dataset_path': "b:/!temp/y_pred_y_true_pairs/pdtsc/train.tfrecord",
        'test_dataset_path': "b:/!temp/y_pred_y_true_pairs/pdtsc/test.tfrecord",
        'cuDNNGRU': False,
        'epochs': 5,
        'batch_size': 64,
        'lr': 0.001,
        'clipnorm': 5.,
        'shuffle_buffer': 200000,
        'checkpoint_dir': "./results/lm/training_ckpt_enc_dec",
        'max_length': 300,
    }

    lm_enc_params = {
        'vocab_size': alphabet_size + 1,  # + 1 for padding value
        'embedding_dim': 32,
        'gru_dims': [32, 64, 32],
    }

    lm_dec_params = {
        'vocab_size': len(c2n_map_lm),
        'embedding_dim': 64,
        'gru_dims': [32, 32, 32],
    }

    # ---------------------

    # Optimizer
    lr = 0.01
    lr_decay = True
    lr_decay_rate = 0.9
    lr_decay_epochs = 2
    epsilon = 0.1
    amsgrad = True

    # Data Augmentation (in pipeline)
    data_aug = {
        'mode': "0x",  # mode of many times to apply data aug (allowed: 0x, 1x or 2x)
        'bandwidth_time': (10, 100),
        'bandwidth_freq': (10, 30),
        'max_percent_time': 0.2,
        'max_percent_freq': 1.,
    }

    # Test Pipeline:
    spell_check = True
    transcribe_digits = True

    # Decoder
    beam_width = 32
    top_paths = 1  # > 1 not implemented

    # Early stopping
    patience_epochs = 3


class PREDICTION_FLAGS(FLAGS):

    recording = {
        "rate": 16000,
        "updates_per_second": 10,
        "channels": 1,
        "max_record_seconds": 30,
    }

    features = {
        "type": "MFSC",
        "energy": True,
        "deltas": (2, 2),
    }

    models = {
        "am_path": "./models/model_combined.h5",
        "lm_path": "bert-base-multilingual-uncased"
    }

    # prediction
    beam_width = 256
    top_paths = 5