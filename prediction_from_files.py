import os
import re

import numpy as np
from matplotlib import pyplot as plt

from DataOps import load_datasets
from FLAGS import PREDICTION_FLAGS
from FeatureExtraction import FeatureExtractor
from Models import save_pred_true_pairs_to_tfrecord, convert_to_strings


_PATHS = ["b:/!temp/PDTSC_MFSC_unigram_40_banks_DEBUG_min_100_max_3000_tfrecord/1.0/"]
_OUTPUT_FOLDER = "b:/!temp/y_pred_y_true_pairs"
_OUTPUT_FILENAMES = ["/train", "/test"]
_OUTPUT_PATHS = ["".join([_OUTPUT_FOLDER, sf]) for sf in _OUTPUT_FILENAMES]
_TRANSCRIBE = True


def read_chunk(stream, chunk_size):
    data_string = stream.read(chunk_size)
    return np.frombuffer(data_string, dtype=np.float32)


def plot_audio(timespan, frames, axes=plt):

    axes.plot(timespan, frames)
    axes.autoscale(enable=True, axis='x', tight=True)
    title_str = "Audio signal"
    xlabel_srt = "Time (s)"
    ylabel_str = "Amplitude (1)"
    if isinstance(axes, plt.Axes):
        axes.set_title(title_str)
        axes.set_xlabel(xlabel_srt)
        axes.set_ylabel(ylabel_str)
    else:
        axes.title(title_str)
        axes.xlabel(xlabel_srt)
        axes.ylabel(ylabel_str)


def mask_sentence(sentence, fill_mask_pipeline):
    words = sentence.split(" ")
    nW = len(words)
    for i in range(nW):
        words[i] = "[MASK]"
        masked_sent = " ".join(words)
        if i+1 == nW:
            masked_sent += "."
        results = fill_mask_pipeline(masked_sent)
        top_sent = re.sub(r"(\[CLS\]|\[SEP\])", "", results[0]["sequence"])
        print(top_sent)

        # TODO: continue

        sentence = top_sent
        words = sentence.split(" ")

    return sentence


if __name__ == '__main__':
    # set logging to only show errors
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    print("INITIALIZING FEATURE EXTRACTOR".center(50, "_"))
    extractor = FeatureExtractor(PREDICTION_FLAGS.recording['rate'],
                                 feature_type=PREDICTION_FLAGS.features['type'],
                                 energy=PREDICTION_FLAGS.features['energy'],
                                 deltas=PREDICTION_FLAGS.features['deltas'])


    for path in _PATHS:
        print(f"Current FOLDER: {os.path.split(path)[-1]}".center(50, "_"))
        ds_train, ds_test, num_train_batches, num_test_batches = load_datasets(path)

        print("SAVING PREDICTIONS FROM MODEL AND TRUE VALUES TO TFRECORDS".center(50, "_"))
        save_pred_true_pairs_to_tfrecord(PREDICTION_FLAGS.models['am_path'],
                                         ds_train,
                                         output_path=_OUTPUT_PATHS[0])
        save_pred_true_pairs_to_tfrecord(PREDICTION_FLAGS.models['am_path'],
                                         ds_test,
                                         output_path=_OUTPUT_PATHS[1])