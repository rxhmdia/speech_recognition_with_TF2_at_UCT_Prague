# https://www.tensorflow.org/tutorials/load_data/text
import os

from DataOps import PDTSCLoader, OralLoader
from helpers import console_logger

LOGGER = console_logger(__name__, "DEBUG")

DATASET = "oral"  # "pdtsc" or "oral"
DEBUG = False

if "pdtsc" in DATASET.lower():
    TRANSCRIPT_FOLDER = "b:/!DATASETS/PDTSC/transcripts"
elif "oral" in DATASET.lower():
    TRANSCRIPT_FOLDER = "b:/!DATASETS/oral2013/transcripts"
else:
    raise ValueError("DATASET should be either pdtsc or oral.")
TRANSCRIPT_FOLDER += "_debug/" if DEBUG else "/"
LABEL_SAVE_FILE_PATH = os.path.join(TRANSCRIPT_FOLDER, "labels.txt")

N2C_MAP = PDTSCLoader.n2c_map
N2C_MAP[13] = "*"  # change "ch" mapping to "*" so that it registers as one character

if __name__ == '__main__':
    audiofiles = []
    _, _, transcripts = next(os.walk(TRANSCRIPT_FOLDER))
    LOGGER.debug(f"transcripts: {transcripts}")
    transcripts_full = [os.path.join(TRANSCRIPT_FOLDER, t) for t in transcripts]
    if "pdtsc" in DATASET.lower():
        loader = PDTSCLoader(audiofiles, transcripts_full)
    elif "oral" in DATASET.lower():
        loader = OralLoader(audiofiles, transcripts_full)
    else:
        raise ValueError("DATASET should be either pdtsc or oral.")

    LOGGER.info("Creating labels!")
    labels = loader.transcripts_to_labels()
    if "oral" in DATASET:
        labels = list(labels.values())
        labels = [[l[0] for lab in labels for l in lab]]
    LOGGER.debug(f"label from first sentence in first file: {labels[0][0]}")
    LOGGER.debug(f"decoded: {loader.num2char([labels[0][0]], N2C_MAP)}")

    LOGGER.info("Decode back into sentences")
    decoded_labels = []
    for lab_file in labels:
        decoded_labels.extend(loader.num2char(lab_file, N2C_MAP))
    LOGGER.debug(f"Decoded labels: {decoded_labels}")

    LOGGER.info("Add lineends at end of each sentence")
    decoded_labels = [l+"\n" for l in decoded_labels]
    LOGGER.debug(f"Decoded labels: {decoded_labels}")

    LOGGER.info("Save sentences to file for further processing.")
    with open(LABEL_SAVE_FILE_PATH, "w") as f:
        f.writelines(decoded_labels)
