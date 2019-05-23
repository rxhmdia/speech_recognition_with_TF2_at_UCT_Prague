# TODO: transform audio to features
    # load audio into FeatureExtractor
    # transform into features
# TODO: load model

import os

import pyaudio
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

from FLAGS import PREDICTION_FLAGS
from FeatureExtraction import FeatureExtractor
from Model import predict_from_saved_model


def read_chunk(stream, chunk_size):
    data_string = stream.read(chunk_size)
    return np.frombuffer(data_string, dtype=np.float32)


def record_audio(record_seconds):
    audio_format = pyaudio.paFloat32
    chunk_size = int(PREDICTION_FLAGS.recording['rate']/PREDICTION_FLAGS.recording['updates_per_second'])
    frames = []

    p = pyaudio.PyAudio()

    stream = p.open(format=audio_format,
                    channels=PREDICTION_FLAGS.recording['channels'],
                    rate=PREDICTION_FLAGS.recording['rate'],
                    input=True,
                    frames_per_buffer=chunk_size)

    for i in range(PREDICTION_FLAGS.recording['updates_per_second']*record_seconds):
        if i % PREDICTION_FLAGS.recording['updates_per_second'] == 0:
            print(str(int(i/PREDICTION_FLAGS.recording['updates_per_second'])) + " s ...")
        audio_chunk = read_chunk(stream, chunk_size)
        frames.extend(audio_chunk)

    timespan = np.arange(0, record_seconds, 1/PREDICTION_FLAGS.recording['rate'])

    return timespan, frames


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


if __name__ == '__main__':
    # set logging to only show errors
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    print("\n_____RECORDING AUDIO_____")
    timespan, frames = record_audio(5)

    print("\n_____CONVERTING TO FEATURE REPRESENTATION_____")
    extractor = FeatureExtractor([np.array(frames)], PREDICTION_FLAGS.recording['rate'],
                                 feature_type=PREDICTION_FLAGS.features['type'],
                                 energy=PREDICTION_FLAGS.features['energy'],
                                 deltas=PREDICTION_FLAGS.features['deltas'])

    features = extractor.transform_data()[0]

    print("\n_____PREDICTING FROM SAVED MODEL_____")
    predict_from_saved_model(PREDICTION_FLAGS.model['path'], features)

    print("\n_____PLOTTING AUDIO AND FEATURES_____")
    fig, ax = plt.subplots(2)
    plot_audio(timespan, frames, axes=ax[0])
    extractor.plot_cepstra([features], 1, axes=ax[1])
    plt.show()





