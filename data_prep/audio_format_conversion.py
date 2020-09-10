import os
import soundfile as sf
import librosa as lb

PDTSC_ROOT = "b:/!DATASETS/PDTSC/audio/"
CV_ROOT = "b:/!DATASETS/CommonVoice/cs/clips/"


def convert_to_flac(load_path):
    signal, fs = sf.read(load_path)
    write_path = f"{os.path.splitext(load_path)[0]}.flac"
    sf.write(write_path, signal, fs)
    print(f"{load_path} converted to {write_path}")


def convert_mp3_to_wav(load_path, fs=16000):
    signal, fs = lb.load(load_path, sr=fs)
    write_path = f"{os.path.splitext(load_path)[0]}.wav"
    sf.write(write_path, signal, fs)
    print(f"{load_path} converted to {write_path}")


if __name__ == '__main__':

    # load_paths_gen = os.walk(PDTSC_ROOT)
    # for root, dirs, load_paths in load_paths_gen:
    #     for path in load_paths:
    #         full_path = os.path.join(root, path)
    #         convert_to_flac(full_path)

    load_paths_gen = os.walk(CV_ROOT)
    for root, dirs, load_paths in load_paths_gen:
        for path in load_paths:
            full_path = os.path.join(root, path)
            convert_mp3_to_wav(full_path)