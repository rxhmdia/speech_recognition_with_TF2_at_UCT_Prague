import logging
import absl


def console_logger(name=__name__, level=logging.WARNING):

    if not isinstance(level, (str, int)):
        raise TypeError("Logging level data type is not recognised. Should be str or int.")

    logger = logging.getLogger(name)

    if logger.hasHandlers():
        logger.handlers.clear()
    logger.setLevel(level)
    formatter = logging.Formatter('%(levelname)7s (%(name)s) %(asctime)s - %(message)s', datefmt="%Y-%m-%d %H:%M:%S")
    if "tensorflow" in name:
        absl.logging.get_absl_handler().setFormatter(formatter)
    else:
        console = logging.StreamHandler()
        console.setLevel(level)
        console.setFormatter(formatter)
        logger.addHandler(console)
    return logger


def extract_channel(signal, channel_number):
    """Extract single channel from a multi-channel (stereo) audio signal"""
    try:
        return signal[:, channel_number]
    except IndexError:
        return signal
