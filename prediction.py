# TODO: transform audio to features
    # load audio into FeatureExtractor
    # transform into features
# TODO: load model

import os
import re

import pyaudio
import numpy as np
from matplotlib import pyplot as plt


from FLAGS import PREDICTION_FLAGS
from FeatureExtraction import FeatureExtractor
from Models import predict_from_saved_model, convert_to_strings

from transformer_support import masked_pipeline_from_trained_model


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
                    output=True,
                    frames_per_buffer=chunk_size)

    for i in range(PREDICTION_FLAGS.recording['updates_per_second']*record_seconds):
        if i % PREDICTION_FLAGS.recording['updates_per_second'] == 0:
            print(str(int(i/PREDICTION_FLAGS.recording['updates_per_second'])) + " s ...")
        audio_chunk = read_chunk(stream, chunk_size)
        frames.extend(audio_chunk)

    timespan = np.arange(0, record_seconds, 1/PREDICTION_FLAGS.recording['rate'])

    return timespan, frames, stream


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

#    print("INITIALIZING LANGUAGE MODEL PIPELINE".center(50, "_")
#    lm_mask_pipeline = masked_pipeline_from_trained_model(PREDICTION_FLAGS.models['lm_path'])

    print("INITIALIZING FEATURE EXTRACTOR".center(50, "_"))
    extractor = FeatureExtractor(PREDICTION_FLAGS.recording['rate'],
                                 feature_type=PREDICTION_FLAGS.features['type'],
                                 energy=PREDICTION_FLAGS.features['energy'],
                                 deltas=PREDICTION_FLAGS.features['deltas'])

    print("RECORDING AUDIO".center(50, "_"))
    timespan, frames, stream = record_audio(5)

    print("CONVERTING TO FEATURE REPRESENTATION".center(50, "_"))
    features = extractor.transform_data([np.array(frames)])[0]

    print("PREDICTING FROM SAVED MODEL".center(50, "_"))
    predictions = predict_from_saved_model(PREDICTION_FLAGS.models['am_path'], features)

    print("TRANSCRIBING TO STRINGS".center(50, "_"))
    string_predictions = convert_to_strings(predictions, apply_autocorrect=True, digitize=True)

#    print("RUNNING THROUGH LANGUAGE MODEL".center(50, "_")
#    sentence = mask_sentence(decoded_predictions[0][2], lm_mask_pipeline)

    print("RESULT".center(50, "_"))
    print(f"AM: {string_predictions[0][2]}")
#    print(f"LM: {sentence}")

    print("REPLAYING AUDIO STREAM".center(50, "_"))
    stream.write(b"".join(frames))

    print("CLOSING STREAM".center(50, "_"))
    stream.stop_stream()
    stream.close()

    print("PLOTTING AUDIO AND FEATURES".center(50, "_"))
    fig, ax = plt.subplots(2)
    plot_audio(timespan, frames, axes=ax[0])
    extractor.plot_cepstra([features], 1, axes=ax[1])
    plt.show()





