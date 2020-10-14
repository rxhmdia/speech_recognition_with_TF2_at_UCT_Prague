from DataOps import CommonVoiceLoader
from FLAGS import FLAGS

from helpers import console_logger

LOGGER = console_logger(__name__, "DEBUG")

if __name__ == '__main__':
    LOGGER.info("INITIALIZING CommonVoiceLoader")
    cvl_loader = CommonVoiceLoader(["b:/!DATASETS/CommonVoice/cs/train.tsv", "b:/!DATASETS/CommonVoice/cs/test.tsv"],
                                   transcribe_digits=True)

    LOGGER.info("CONVERTING TRANSCRIPTS TO LABELS")
    labels = cvl_loader.transcripts_to_labels()

    LOGGER.info("DECODING A SAMPLE SENTENCE")
    decoded = cvl_loader.num2char([labels["train"][0]], FLAGS.n2c_map)
    LOGGER.debug(f"decoded: {decoded}")
    LOGGER.debug(f"audiofiles: {cvl_loader.audiofiles}")

    LOGGER.info("PRINTING NONZERO COUNTS OF DIGIT REPLACEMENTS")
    cvl_loader.dt.print_nonzero_counts()

    LOGGER.info("LOADING AUDIO")
    audio, fs = cvl_loader.load_audio()
    LOGGER.debug(f"|SAMPLE| audio: {audio['train'][0]} | fs: {fs['train']}")