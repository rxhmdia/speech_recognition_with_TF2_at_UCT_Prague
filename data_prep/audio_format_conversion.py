import os
import soundfile as sf

ROOT = "b:/!DATASETS/PDTSC/audio/"


def convert_to_flac(load_path):
    signal, fs = sf.read(load_path)
    write_path = f"{os.path.splitext(load_path)[0]}.flac"
    sf.write(write_path, signal, fs)
    print(f"{load_path} converted to {write_path}")


if __name__ == '__main__':
    load_paths_gen = os.walk(ROOT)

    for root, dirs, load_paths in load_paths_gen:
        for path in load_paths:
            full_path = os.path.join(root, path)
            convert_to_flac(full_path)