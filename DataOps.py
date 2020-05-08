import json
import re
import os
import shutil

from copy import deepcopy
from math import factorial
from itertools import compress

# from icu import LocaleData
import numpy as np
import tensorflow as tf
import soundfile as sf  # for loading audio in various formats (OGG, WAV, FLAC, ...)

from bs4 import BeautifulSoup
from pysndfx import AudioEffectsChain

from FeatureExtraction import FeatureExtractor
from FLAGS import FLAGS
from helpers import console_logger, extract_channel, if_bool, if_float, if_int, if_str

LOGGER = console_logger(__name__, FLAGS.logger_level)
_AUTOTUNE = tf.data.experimental.AUTOTUNE


"""#####################################################################################################################
### |                                                                                                              | ###
### |--------------------------------------------------DATA LOADER-------------------------------------------------| ###
### |                                                                                                              | ###
#####################################################################################################################"""


class DataLoader:
    c2n_map = FLAGS.c2n_map
    n2c_map = FLAGS.n2c_map

    def __init__(self, audiofiles, transcripts, bigrams=False, repeated=False):
        """ Initialize DataLoader() object

        :param audiofiles: List[paths] to audio files
        :param transcripts: List[paths] to transcripts of the audio files
        :param bigrams: (bool) whether the labels should be made into bigrams or not
        :param repeated: (bool) whether the bigrams should contain repeated characters (eg: 'aa', 'bb')
        """
        self.audiofiles = audiofiles
        self.transcripts = transcripts
        self.bigrams = bigrams
        self.repeated = repeated
        self.audio = [[np.array(0, dtype=np.float32)]]*len(self.audiofiles)
        """
        :ivar fs: (int) sampling rates of loaded audio files
        :ivar starts: (List[List[float]]) starting times of the sentences
        :ivar ends: (List[List[float]]) ending times of the sentences
        :ivar tokens: (List[List[str]]) tokens (sentences) from transcripts
        :ivar labels: (List[nDArr]) numeric representation of tokens (char -> int)
        """
        self.fs = np.zeros((len(self.audiofiles)), dtype=np.uint16)
        self.starts = [np.array(0, dtype=np.float32)]*len(self.transcripts)
        self.ends = [np.array(0, dtype=np.float32)]*len(self.transcripts)
        self.tokens = [[]]*len(self.transcripts)
        self.labels = [[np.array(0, dtype=np.int32)]]*len(self.transcripts)

        if self.bigrams:
            self.b2n_map = self.calc_bigram_map(self.c2n_map, repeated=self.repeated)
            self.n2b_map = self.calc_bigram_map(self.n2c_map, repeated=self.repeated)

    @staticmethod
    def char2num(sentlist, c2n_map):
        """ Transform list of sentences (tokens) to list of lists with numeric representations of the
        characters depending on their position in the czech alphabet.
        """
        arraylist = [np.asarray([c2n_map[c] if c in c2n_map.keys() else c2n_map[' ']
                                 for c in chars.lower()], dtype=np.int32) for chars in sentlist]
        for i in range(len(arraylist)):
            ch_idcs = [(r.start(), r.end() - 1) for r in re.finditer('ch', sentlist[i])]

            # change arraylist at ch_idcs starts to number for symbol 'ch'
            mask_change = np.zeros(len(arraylist[i]), dtype=bool)
            mask_change[[tup[0] for tup in ch_idcs]] = True
            arraylist[i][mask_change] = c2n_map['ch']

            # remove elements after added numbers for symbol 'ch'
            mask_delete = np.ones(len(arraylist[i]), dtype=bool)
            mask_delete[[tup[1] for tup in ch_idcs]] = False
            arraylist[i] = arraylist[i][mask_delete]

        return arraylist

    @staticmethod
    def bigram2num(bigramlist, b2n_map):
        """ Transform list of lists of bigrams to numeric values"""
        return [[b2n_map[bigram] for bigram in token] for token in bigramlist]

    @staticmethod
    def num2char(arraylist, n2c_map):
        """ Transform list of numpy arrays with chacater/bigram numbers to list of sentences"""
        return [''.join([n2c_map[o] for o in arr]) for arr in arraylist]

    @staticmethod
    def k_perms_of_n(k, n, repeated=True):
        if repeated:
            return n**k
        else:
            if k <= n:
                return factorial(n) // factorial(n - k)
            else:
                raise ValueError("When repeated==False, k must be less than or equal to n")

    @staticmethod
    def calc_bigram_map(input_dict, repeated=True):
        """calculate non-overlapping bigrams from input_dict of n elements

        :param input_dict: dictionary with types {str: int} or {int: str}
        :param repeated: if True, include bigrams with duplicate entries (eg. 'aa', 'bb')
        """

        dict_keys = list(input_dict.keys())
        dict_vals = list(input_dict.values())

        key_types = [type(dict_keys[i]) for i in range(len(input_dict))]
        val_types = [type(dict_vals[i]) for i in range(len(input_dict))]

        if all(key_types[i] == key_types[0] for i in range(len(input_dict))):
            key_type = key_types[0]
        else:
            raise TypeError("Key types are not consistent.")

        if all(val_types[i] == val_types[0] for i in range(len(input_dict))):
            val_type = val_types[0]
        else:
            raise TypeError("Value types are not consistent.")

        if key_type == str and val_type == int:
            dct = {v: k for k, v in input_dict.items()}  # switch the types to {int: str}
        elif key_type == int and val_type == str:
            dct = deepcopy(input_dict)  # create deepcopy with types {int: str}
        else:
            raise TypeError("Keys and Values of the input_dict must have dtypes {int: str} or {str: int}.")

        # remove space character from dct
        space_key = list(dct.keys())[list(dct.values()).index(' ')]
        dct.pop(space_key)

        vocab_size = len(dct)

        outputs = list(dct.values())

        for i in range(vocab_size):
            if repeated:
                other_vals = set(dct.values())
            else:
                other_vals = [val for key, val in dct.items() if key != i]
            outputs.extend([dct[i] + val for val in other_vals])

        # add space character at the end of the new dict
        outputs.extend(' ')

        bigram_size = DataLoader.k_perms_of_n(2, vocab_size, repeated=repeated)
        assert len(outputs) == bigram_size + vocab_size + 1, \
            'Number of entries in output dictionary is not consistent with expected number of bigrams'

        # remove duplicate entry of 'ch'
        outputs.pop(outputs.index('ch', vocab_size))

        out_dict = dict(enumerate(outputs))

        if key_type == str:
            return {v: k for k, v in out_dict.items()}
        else:
            return out_dict

    @staticmethod
    def tokens_to_bigrams(tokens):
        bigram_list = []
        for token in tokens:
            bigrams = []
            for word in token.split(' '):
                # temporarily replace 'ch' with '*' so that it counts as one character
                word = re.sub('ch', '*', word)
                word_len = len(word)
                bigrams.extend([word[i - 1] + word[i] for i in range(1, word_len, 2)])
                if word_len % 2:
                    bigrams.append(word[-1])
                bigrams.append(' ')
            bigrams.pop()
            # turn '*' chars back to 'ch'
            for i in range(len(bigrams)):
                bigrams[i] = re.sub('\*', 'ch', bigrams[i])
            bigram_list.append(bigrams)
        return bigram_list

    @staticmethod
    def load_labels(path_to_files='./data'):
        """ Load labels of transcripts from transcript-###.npy files in specified folder
        into a list of lists of 1D numpy arrays
        :param path_to_files: string path leading to the folder with transcript files or .npy trascript file

        :return list of lists of 1D numpy arrays, list of lists of strings with paths to files
        """

        ext = os.path.splitext(path_to_files)[1]

        # if path_to_files leads to a single (.npy) file , load only the one file
        if ext == ".npy":
            labels = [[np.load(path_to_files)]]
            path_list = [[os.path.abspath(path_to_files)]]
        elif not ext:
            # if the path_to_files contains subfolders, load data from all subfolders
            labels = []
            path_list = []
            subfolders = [os.path.join(path_to_files, subfolder) for subfolder in next(os.walk(path_to_files))[1]]

            # if there are no subfolders in the provided path_to_files, look directly in path_to_files
            if not subfolders:
                subfolders.append(path_to_files)

            for sub in subfolders:
                files = [os.path.splitext(f) for f in os.listdir(sub) if
                         os.path.isfile(os.path.join(sub, f))]
                paths = [os.path.abspath(os.path.join(sub, ''.join(file)))
                         for file in files if 'transcript' in file[0] and file[-1] == '.npy']
                sublabels = [np.load(path) for path in paths]
                path_list.append(paths)
                labels.append(sublabels)
        else:
            raise IOError("Specified file doesn't have .npy suffix.")

        return labels, path_list


class PDTSCLoader(DataLoader):

    def __init__(self, audiofiles, transcripts, bigrams=False, repeated=False):
        super().__init__(audiofiles, transcripts, bigrams, repeated)

    @staticmethod
    def time2secms(timelist):
        """Convert list of times in format hh:mm:ss.ms to numpy array of ss.ms format"""

        ssmsarray = np.zeros_like(timelist, dtype=np.float32)

        for i, time in enumerate(timelist):
            (hh, mm, ssms) = np.float32(time.split(':'))

            ssmsarray[i] = hh*3600 + mm*60 + ssms

        return ssmsarray

    def transcripts_to_labels(self):
        """

        :return: list of lists of transcripts
        """
        for i, file in enumerate(self.transcripts):
            with open(file, 'r', encoding='utf8') as f:
                raw = f.read()

            soup = BeautifulSoup(raw, 'xml')

            # extract relevant tags from the soup
            lm_tags = soup.find_all(lambda tag: tag.name == 'LM' and tag.has_attr('id'))
            start_time_tags = [LM.find('start_time') for LM in lm_tags]
            end_time_tags = [LM.find('end_time') for LM in lm_tags]
            token_tags = [LM.find_all('token') for LM in lm_tags]

            # process the tokens from token tags
            regexp = r'[^A-Za-záéíóúýčďěňřšťůž]+'  # find all non alphabetic characters (Czech alphabet)
            tokens = [' '.join([re.sub(regexp, '', token.text.lower()) for token in tokens])
                      for tokens in token_tags]  # joining sentences and removing special and numeric chars

            empty_idcs = [i for i, token in enumerate(tokens) if not token]  # getting indices of empty tokens

            # removing empty_idcs from starts, ends and tokens
            start_time_tags = [tag for i, tag in enumerate(start_time_tags) if i not in empty_idcs]
            end_time_tags = [tag for i, tag in enumerate(end_time_tags) if i not in empty_idcs]
            tokens = [token for i, token in enumerate(tokens) if i not in empty_idcs]

            # save the start times, ent times and tokens to instance variables
            self.starts[i] = self.time2secms([start.text for start in start_time_tags])
            self.ends[i] = self.time2secms([end.text for end in end_time_tags])
            self.tokens[i] = tokens

            assert len(self.starts[i]) == len(self.ends[i]), "start times and end times don't have the same length"
            assert len(self.ends[i]) == len(self.tokens[i]), "there is different number of tokens than end times"

            # convert characters in tokens to numeric values representing their position in the czech alphabet
            if self.bigrams:
                bigrams = self.tokens_to_bigrams(tokens)
                self.labels[i] = self.bigram2num(bigrams, self.b2n_map)
            else:  # labels are unigrams
                self.labels[i] = self.char2num(tokens, self.c2n_map)

        return self.labels

    def save_labels(self, labels=None, folder='./data/', exist_ok=False):
        """
        Save labels of transcripts to specified folder under folders with names equal to name of the transcrips files
        """
        if not labels:
            if self.labels:
                labels = self.labels
            else:
                print('No labels were given and the class labels have not been generated yet.'
                      'Please call transcripts_to_labels class function first.')
                return

        # get names of the loaded transcript files and use them as subfolder names
        subfolders = tuple(os.path.splitext(os.path.basename(transcript))[0] for transcript in self.transcripts)

        try:
            for subfolder in subfolders:
                os.makedirs(os.path.join(folder, subfolder), exist_ok=exist_ok)
        except OSError:
            print('Subfolders already exist. Please set exist_ok to True if you want to save into them anyway.')
            return

        for idx in range(len(labels)):
            ndigits = len(str(len(labels[idx])))  # zeroes to pad the name with in order to keep the correct order
            fullpath = os.path.join(folder, subfolders[idx])
            for i, array in enumerate(labels[idx]):
                np.save('{0}/transcript-{1:0{2}d}.npy'.format(fullpath, i, ndigits), array)

    def load_audio(self):
        for i, file in enumerate(self.audiofiles):
            signal, self.fs[i] = sf.read(file)

            signal = extract_channel(signal, 0)  # convert signal from stereo to mono by extracting channel 0

            tstart = 0
            tend = signal.shape[0]/self.fs[i]
            tstep = 1/self.fs[i]
            tspan = np.arange(tstart, tend, tstep, dtype=np.float32)

            # find indices corresponding to the start and end times of the transcriptions
            starts_idcs = np.asarray([np.searchsorted(tspan, start) for start in self.starts[i]], dtype=np.int32)
            ends_idcs = np.asarray([np.searchsorted(tspan, end) for end in self.ends[i]], dtype=np.int32)

            # split the signal to intervals (starts_idcs[j], ends_idcs[j])
            # self.audio[i] = [signal[st_idx:ed_idx] for st_idx, ed_idx in zip(starts_idcs, ends_idcs)]
            self.audio[i] = [signal[starts_idcs[j]:ends_idcs[j]] for j in range(starts_idcs.shape[0])]

        return self.audio, self.fs

    @staticmethod
    def save_audio(file, audio, fs):
        sf.write(file, audio, fs)


class OralLoader(DataLoader):
    c2n_map = DataLoader.c2n_map
    c2n_map['*'] = 13  # ch character is subbed as * but in the reverse map it is still 'ch'

    def __init__(self, audiofiles, transcripts, bigrams=False, repeated=False):
        super().__init__(audiofiles, transcripts, bigrams, repeated)
        self.labels = None
        self.audio = dict()  # audiofile dictionary with filenames as keys
        self.fs = dict()  # sampling frequencies of the audiofiles with filenames as keys

    def transcripts_to_labels(self, label_max_duration=10.0):
        turn_info = [list() for _ in range(len(self.transcripts))]
        turn_info_no_overlap = [list() for _ in range(len(self.transcripts))]
        reg_ch = r'ch'  # any sequence of characters 'ch' ... which counts as a single character in Czech
        reg_pthses = r'\(.*?\)'  # any character between parentheses () -- in oral it marks special sounds (laugh, ambient, ...)
        reg_not_czech = r'[^A-Za-záéíóúýčďěňřšťůž ]+'  # all nonalphabetic characters (czech alphabet)
        labels = dict()
        for idx, file in enumerate(self.transcripts):
            with open(file, 'r', encoding='cp1250') as f:
                raw = f.read()

            soup = BeautifulSoup(raw, 'lxml')

            file_name = soup.find('trans')['audio_filename']
            speakers = {s['id'] for s in
                        soup.find_all('speaker')}  # speaker id's that appear in the current transcript file
            turns = soup.find_all('turn')

            # extract time_spans and number of speakers from each turn
            for i, t in enumerate(turns):
                sync_times = [float(sync['time']) for sync in t.find_all('sync')]
                turn_info[idx].append({
                    'sync_times': list(zip(sync_times, [*sync_times[1:], float(t['endtime'])])),
                    'speakers': t['speaker'].split(' ') if 'speaker' in t.attrs.keys() else [],
                    'text': [' '.join(re.sub(reg_not_czech, '',  # remove nonalphabetic characters
                                             re.sub(reg_pthses, '',
                                                    # remove anything in parentheses including the parentheses
                                                    txt.lower())).split()) for txt in t.text[:-1].split('\n\n')[1:]]
                })

            # GET TURNS WITH EXACTLY 1 SPEAKER (removes overlap and ambient noises)
            # create mask in which True means that there is exactly 1 speaker
            one_speaker_mask = [len(turn['speakers']) == 1 for turn in turn_info[idx]]
            num_removed = len(one_speaker_mask) - sum(one_speaker_mask)
            # compress the lists using the mask
            turns_no_overlap = list(compress(turns, one_speaker_mask))
            turn_info_no_overlap[idx] = list(compress(turn_info[idx], one_speaker_mask))

            assert len(turns_no_overlap) == len(turns) - num_removed
            assert len(turn_info_no_overlap[idx]) == len(turn_info[idx]) - num_removed
            assert all([len(t['speakers']) == 1 for t in turn_info_no_overlap[idx]])

            sents = []
            starts = []
            ends = []
            for info in turn_info_no_overlap[idx]:
                sync_times, speaker, text = info.values()
                assert len(sync_times) == len(text)
                # fill sents, starts and ends with the first entries in turn_info
                starts.append(sync_times[0][0])
                ends.append(sync_times[0][1])
                sent_duration = sync_times[0][1] - sync_times[0][0]
                sents.append(text[0])
                for i in range(1, len(sync_times)):
                    utterance_duration = sync_times[i][1] - sync_times[i][0]
                    if sent_duration + utterance_duration < label_max_duration:
                        ends[-1] = sync_times[i][1]
                        sent_duration += utterance_duration
                        sents[-1] += ' ' + text[i]
                    else:
                        starts.append(sync_times[i][0])
                        ends.append(sync_times[i][1])
                        sent_duration = sync_times[i][1] - sync_times[i][0]
                        sents.append(text[i])

            # convert the sentences into integer arrays
            sents = [np.array([self.c2n_map[c] for c in re.sub(reg_ch, '*', s)]) for s in sents]
            labels[file_name] = tuple(zip(sents, starts, ends))

        self.labels = labels

        return labels

    def load_audio(self):
        for i, file in enumerate(self.audiofiles):
            path, filename = os.path.split(file)
            filename, ext = os.path.splitext(filename)

            signal, fs = sf.read(file)

            # create array with sampling times of the audiofile
            tstart = 0
            tend = signal.shape[0] / fs
            tstep = 1 / fs
            tspan = np.arange(tstart, tend, tstep, dtype=np.float32)

            starts = []
            ends = []
            for label in self.labels[filename]:
                starts.append(label[1])
                ends.append(label[2])

            starts_idcs = np.asarray([np.searchsorted(tspan, start) for start in starts], dtype=np.int32)
            ends_idcs = np.asarray([np.searchsorted(tspan, end) for end in ends], dtype=np.int32)

            self.audio[filename] = [signal[starts_idcs[j]:ends_idcs[j]] for j in range(starts_idcs.shape[0])]
            self.fs[filename] = fs

        return self.audio, self.fs

    @staticmethod
    def save_audio(file, audio, fs):
        sf.write(file, audio, fs)

    def save_labels(self, labels=None, folder='./data/oral2013/', exist_ok=False):
        """
        Save labels of transcripts to specified folder under folders with names equal to name of the transcrips files
        """
        if not labels:
            if self.labels:
                labels = self.labels
            else:
                print('No labels were given and the class labels have not been generated yet.'
                      'Please call transcripts_to_labels class function first.')
                return

        subfolders = tuple(labels.keys())

        try:
            for subfolder in subfolders:
                os.makedirs(os.path.join(folder, subfolder), exist_ok=exist_ok)
        except OSError:
            print('Subfolders already exist. Please set exist_ok to True if you want to save into them anyway.')
            return

        for key, vals in labels.items():
            ndigits = len(str(len(vals)))
            fullpath = os.path.join(folder, key)
            for i, (sent, _, _) in enumerate(vals):
                np.save('{0}/transcript-{1:0{2}d}.npy'.format(fullpath, i, ndigits), sent)
            print(key + ' saved to ' + fullpath)

    @staticmethod
    def load_labels(path_to_files='./data'):
        """ Load labels of transcripts from transcript-###.npy files in specified folder
        into a dictionary of labels and paths to their files
        :param path_to_files: string path leading to the folder with transcript files or .npy trascript file

        :return Dict["folder/file_name":Tuple[List[labels], List[path_to_files]]]
        """

        ext = os.path.splitext(path_to_files)[1]

        # if path_to_files leads to a single (.npy) file , load only the one file
        if ext == ".npy":
            key = os.path.splitext(os.path.basename(path_to_files))[0]
            labels = {key: ([np.load(path_to_files)],
                            [os.path.abspath(path_to_files)])}
        elif not ext:
            # if the path_to_files contains subfolders, load data from all subfolders
            labels = dict()
            path_list = []
            subfolders = [os.path.join(path_to_files, subfolder) for subfolder in next(os.walk(path_to_files))[1]]

            # if there are no subfolders in the provided path_to_files, look directly in path_to_files
            if not subfolders:
                subfolders.append(path_to_files)

            for sub in subfolders:
                files = [os.path.splitext(f) for f in os.listdir(sub) if
                         os.path.isfile(os.path.join(sub, f))]
                paths = [os.path.abspath(os.path.join(sub, ''.join(file)))
                         for file in files if 'transcript' in file[0] and file[-1] == '.npy']
                sublabels = [np.load(path) for path in paths]
                labels[os.path.normpath(sub).split('\\')[-1]] = tuple(zip(sublabels, paths))
        else:
            raise IOError("Specified file doesn't have .npy suffix.")

        return labels


"""#####################################################################################################################
### |                                                                                                              | ###
### |-------------------------------------------------DATA PIPELINE------------------------------------------------| ###
### |                                                                                                              | ###
#####################################################################################################################"""

# TODO:
#  AdditiveNoise?
# DONE:
#  SpecAug doesn't work with None shapes (so for time masking, the current code doesn't work)
#  Batching only works if the explicit shapes in batch are all the same (freq mask works only without random bandwidth)
#  Ensure resulting frequency shape is the same before and after masking? (pad with zeros instead of removing)
#  TimeMasking
#  FrequencyMasking
#  multiple instances of SpecAug (2x TimeMasking, 2x TimeMasking)
#  make SpecAug mask bandwidths randomly generated!
#  add parameter 'p in (0, 1)' for determining maximum length of time masking relative to time length of current signal


# noinspection DuplicatedCode
class SpecAug:
    _axis_default = 0
    _bandwidth_default = (0, 20)
    _max_percent_default = 1.0

    def __init__(self, axis=_axis_default, bandwidth=_bandwidth_default, max_percent=_max_percent_default):
        """ Tensorflow data pipeline implementation of SpecAug time and frequency masking

        :param axis (int): which axis will be masked (0 ... time, 1 ... frequency)
        :param bandwidth Tuple(int>0, int>0): min and max length of the masked area
        :param max_percent (float): value between (0, 1] maximum ratio of length of bandwidth to length of signal
        """
        self.axis = axis if axis in (0, 1) else self._axis_default
        self.bandwidth = bandwidth
        self.max_percent = max_percent if 0. < max_percent <= 1. else self._max_percent_default
        self._max_sx = None

    @tf.function(experimental_relax_shapes=True)
    def _mask_sample(self, sample, sx_max=None):
        x, y, sx, sy = sample
        stime, sfreq = (sx, x.shape[1])

        if self.axis == 0:
            full_len = stime
        elif self.axis == 1:
            full_len = sfreq
        else:
            raise AttributeError("self.axis must be either 0 (time masking) or 1 (frequency masking)")

        # generate position of masking
        mbw = int(float(full_len)*self.max_percent)  # maximum bandwidth based on length of signal
        max_bandwidth = self.bandwidth[1] if mbw > self.bandwidth[1] else mbw
        min_bandwidth = self.bandwidth[0] if max_bandwidth > self.bandwidth[0] else max_bandwidth-1
        bandwidth = tf.random.uniform([], min_bandwidth, max_bandwidth, dtype=tf.int32)  # random length of mask
        tm_lb = tf.random.uniform([], 0, full_len - bandwidth, dtype=tf.int32)  # lower bounds
        tm_ub = tm_lb + bandwidth  # upper bounds

        # generate lower bound and upper bound masks
        mask_lb = tf.concat((tf.ones([tm_lb, ], dtype=tf.bool), tf.zeros([full_len - tm_lb, ], dtype=tf.bool)), axis=0)
        mask_ub = tf.concat((tf.zeros([tm_ub, ], dtype=tf.bool), tf.ones([full_len - tm_ub], dtype=tf.bool)), axis=0)

        # get value for padding batch to same time length
        padding = sx_max - stime

        if self.axis == 0:
            # TIME MASKING
            x = tf.concat((tf.boolean_mask(x, mask_lb, axis=0),
                           tf.zeros([bandwidth + padding, sfreq]),
                           tf.boolean_mask(x, mask_ub, axis=0)), axis=0)
        elif self.axis == 1:
            # FREQUENCY MASKING
            # x = tf.pad(x, [[0, padding], [0, 0]])
            x = tf.concat((tf.boolean_mask(x, mask_lb, axis=1),
                           tf.zeros([sx_max, bandwidth]),
                           tf.boolean_mask(x, mask_ub, axis=1)), axis=1)
        else:
            raise AttributeError("self.axis must be either 0 (time masking) or 1 (frequency masking)")

        # sx = sx + padding
        x = tf.ensure_shape(x, (None, sfreq))

        return x, y, sx, sy

    @tf.function(experimental_relax_shapes=True)
    def mask(self, x, y, sx, sy):
        return tf.map_fn(lambda sample: self._mask_sample(sample, tf.reduce_max(sx)),
                         (x, y, sx, sy),
                         parallel_iterations=4)


def _parse_proto(example_proto):
    features = {
        'x': tf.io.FixedLenSequenceFeature([FLAGS.num_features], tf.float32, allow_missing=True),
        'y': tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
    }
    parsed_features = tf.io.parse_single_example(example_proto, features)
    return parsed_features['x'], parsed_features['y']


def _read_tfrecords(file_names=("file1.tfrecord", "file2.tfrecord", "file3.tfrecord"),
                    shuffle=False, seed=None, block_length=FLAGS.num_train_data, cycle_length=8):
    files = tf.data.Dataset.list_files(file_names, shuffle=shuffle, seed=seed)
    ds = files.interleave(lambda x: tf.data.TFRecordDataset(x).map(_parse_proto,
                                                                   num_parallel_calls=_AUTOTUNE),
                          block_length=block_length,
                          cycle_length=cycle_length,
                          num_parallel_calls=_AUTOTUNE)
    ds = ds.map(lambda x, y: (x, y, tf.shape(x)[0], tf.size(y)), num_parallel_calls=_AUTOTUNE)
    return ds


def _bucket_and_batch(ds, bucket_boundaries):
    num_buckets = len(bucket_boundaries) + 1
    bucket_batch_sizes = [FLAGS.batch_size_per_GPU] * num_buckets
    padded_shapes = (tf.TensorShape([None, FLAGS.num_features]),  # cepstra padded to maximum time in batch
                     tf.TensorShape([None]),  # labels padded to maximum length in batch
                     tf.TensorShape([]),  # sizes not padded
                     tf.TensorShape([]))  # sizes not padded
    padding_values = (tf.constant(FLAGS.feature_pad_val, dtype=tf.float32),  # cepstra padded with feature_pad_val
                      tf.constant(FLAGS.label_pad_val, dtype=tf.int64),  # labels padded with label_pad_val
                      0,  # size(cepstrum) -- unused
                      0)  # size(label) -- unused

    bucket_transformation = tf.data.experimental.bucket_by_sequence_length(
        element_length_func=lambda x, y, size_x, size_y: size_x,
        bucket_boundaries=bucket_boundaries,
        bucket_batch_sizes=bucket_batch_sizes,
        padded_shapes=padded_shapes,
        padding_values=padding_values
    )

    ds = ds.apply(bucket_transformation)
    return ds


# noinspection PyShadowingNames
def load_datasets(load_dir,
                  data_aug=FLAGS.data_aug['mode'],
                  bandwidth_time=FLAGS.data_aug['bandwidth_time'],
                  bandwidth_freq=FLAGS.data_aug['bandwidth_freq'],
                  max_percent_time=FLAGS.data_aug['max_percent_time'],
                  max_percent_freq=FLAGS.data_aug['max_percent_freq']):
    path_gen = os.walk(load_dir)

    ds_train = None
    ds_test = None

    if '1x' in data_aug or '2x' in data_aug:
        LOGGER.info("Initializing Data Augmentation for time and freq.")
        sa_time = SpecAug(axis=0, bandwidth=bandwidth_time, max_percent=max_percent_time)
        sa_freq = SpecAug(axis=1, bandwidth=bandwidth_freq, max_percent=max_percent_freq)
        LOGGER.debug(f"sa_time.bandwidth: {sa_time.bandwidth} |"
                     f"sa_freq.bandwidth: {sa_freq.bandwidth}")
    else:
        LOGGER.info("Skipping Pipeline Data Augmentation.")
        sa_time = None
        sa_freq = None

    # load datasets from .tfrecord files in test and train folders
    for path, subfolders, files in path_gen:
        folder_name = os.path.split(path)[-1]
        files = [f for f in files if '.tfrecord' in f]
        fullpaths = [os.path.join(path, f) for f in files]
        if folder_name == '' and len(files) > 0:
            num_data = FLAGS.num_train_data + FLAGS.num_test_data
            ds = _read_tfrecords(fullpaths, shuffle=True, seed=FLAGS.shuffle_seed, block_length=num_data,
                                 cycle_length=1)
            ds = ds.shuffle(FLAGS.num_train_data + FLAGS.num_test_data, seed=FLAGS.shuffle_seed,
                            reshuffle_each_iteration=False)
            ds_train = ds.take(FLAGS.num_train_data)
            ds_test = ds.skip(FLAGS.num_train_data)
            LOGGER.info(f"joined dataset loaded from {path} and split into ds_train ({FLAGS.num_train_data}) "
                        f"and ds_test (rest)")
            break
        if folder_name == 'test':
            ds_test = _read_tfrecords(fullpaths, block_length=FLAGS.num_test_data)
            LOGGER.info(f'test dataset loaded from {path}')
        elif folder_name == 'train':
            # don't shuffle if using shards, because bucketting doesn't work well with shards
            ds_train = _read_tfrecords(fullpaths, block_length=FLAGS.num_train_data)
            LOGGER.info(f'train dataset loaded from {path}')
        else:
            continue

    # BUCKET AND BATCH DATASET
    bucket_boundaries = list(range(FLAGS.min_time, FLAGS.max_time + 1, FLAGS.bucket_width))
    num_buckets = len(bucket_boundaries) + 1
    num_train_batches = (np.ceil(FLAGS.num_train_data / FLAGS.batch_size_per_GPU) + num_buckets).astype(np.int32)
    num_test_batches = (np.ceil(FLAGS.num_test_data / FLAGS.batch_size_per_GPU) + num_buckets).astype(np.int32)

    # train dataset
    ds_train = _bucket_and_batch(ds_train,
                                 bucket_boundaries)  # convert ds into batches of simmilar length features (bucketed)
    # DATA AUGMENTATION
    if '2x' in data_aug:
        ds_train = (ds_train.map(sa_time.mask, num_parallel_calls=_AUTOTUNE)   # time masking 1
                            .map(sa_time.mask, num_parallel_calls=_AUTOTUNE)   # time masking 2
                            .map(sa_freq.mask, num_parallel_calls=_AUTOTUNE)   # frequency masking 1
                            .map(sa_freq.mask, num_parallel_calls=_AUTOTUNE))  # frequency masking 2
    elif '1x' in data_aug:
        ds_train = (ds_train.map(sa_time.mask, num_parallel_calls=_AUTOTUNE)  # time masking
                            .map(sa_freq.mask, num_parallel_calls=_AUTOTUNE))  # frequency masking
    else:
        LOGGER.info("Data Augmentation NOT added into pipeline.")
    ds_train = ds_train.shuffle(buffer_size=FLAGS.buffer_size,
                                reshuffle_each_iteration=True)
    ds_train = ds_train.prefetch(_AUTOTUNE)

    # test dataset
    ds_test = _bucket_and_batch(ds_test, bucket_boundaries)
    ds_test = ds_test.prefetch(_AUTOTUNE)

    return ds_train, ds_test, num_train_batches, num_test_batches


"""#####################################################################################################################
### |                                                                                                              | ###
### |-----------------------------------------------DATA PREPARATION-----------------------------------------------| ###
### |                                                                                                              | ###
#####################################################################################################################"""


class DataPrep:
    # allowed and default values for init
    __datasets = ("pdtsc", "oral")
    __feature_types = ("MFSC", "MFCC")
    __label_types = ("unigram", "bigram")
    __repeated = False
    __energy = True
    __deltas = (2, 2)
    __nbanks = 40
    __filter_nan = True
    __sort = False
    __label_max_duration = 10.0
    __speeds = (1.0, )
    __min_frame_length = 100
    __max_frame_length = 3000
    __modes = ('copy', 'move')
    __delete_unused = False
    __feature_names = 'cepstrum'
    __label_names = 'transcript'
    __tt_split_ratio = 0.9
    __train_shard_size = 2**10
    __test_shard_size = 2**7
    __delete_converted = False
    __debug = False

    def __init__(self, audio_folder, transcript_folder, save_folder, dataset=__datasets[0],
                 feature_type=__feature_types[0], label_type=__label_types[0], repeated=__repeated,
                 energy=__energy, deltas=__deltas, nbanks=__nbanks, filter_nan=__filter_nan, sort=__sort,
                 label_max_duration=10.0, speeds=(1.0, ), min_frame_length=__min_frame_length,
                 max_frame_length=__max_frame_length, mode=__modes[0], delete_unused=__delete_unused,
                 feature_names=__feature_names, label_names=__label_names, tt_split_ratio=__tt_split_ratio,
                 train_shard_size=__train_shard_size, test_shard_size=__test_shard_size,
                 delete_converted=__delete_converted, debug=__debug):
        """ End-to-end data preparation of raw features and labels into tfrecord files ready to be fed into the AM

        :param audio_folder (string): path to folder with raw audio files (.wav or .ogg)
        :param transcript_folder (string): path to folder with raw transcript files (.txt)
        :param save_folder (string): path to folder in which to save the preprocessed data
        :param dataset (string): which dataset is to be expected (allowed:"pdtsc" or "oral")
        :param feature_type (string): which feature type should the data be converted to (allowed: "MFSC" or "MFCC")
        :param label_type (string): type of labels (so far only "unigram" is implemented)
        :param repeated (bool): whether the bigrams should contain repeated characters (eg: 'aa', 'bb')
        :param energy (bool): whether energy feature should be included into feature matrix
        :param deltas (Tuple[int, int]): area from which to calculate differences for deltas and delta-deltas
        :param nbanks (int): number of mel-scaled filter banks
        :param filter_nan (bool): whether to filter-out inputs with NaN values
        :param sort (bool): whether to sort resulting cepstra by file size (i.e. audio length)
        :param label_max_duration (float): maximum time duration of the audio utterances
        :param speeds (Tuple[float, ...]): speed augmentation multipliers (value between 0. and 1.)
        :param min_frame_length (int): signals with less time-frames will be excluded
        :param max_frame_length (int): signals with more time-frames will be excluded
        :param mode (string): whether to copy or move the not excluded files to a new folder
        :param delete_unused (bool): whether to delete files that were unused in the final dataset
        :param feature_names (string): part of filename that all feature files have in common
        :param label_names (string): part of filename that all label files have in common
        :param tt_split_ratio (float): split ratio of training and testing data files (value between 0. and 1.)
        :param train_shard_size (int): approximate tfrecord shard sizes for training data (in MB)
        :param test_shard_size (int): approximate tfrecord shard sizes for testing data (in MB)
        :param delete_converted (bool): whether to delete .npy shard folders that were already converted to .tfrecords
        :param debug (bool): switch between normal and debug mode
        """

        # 01_prepare_data params
        self.audio_folder = os.path.normpath(if_str(audio_folder, "audio_folder"))
        self.transcript_folder = os.path.normpath(if_str(transcript_folder, "transcript_folder"))
        self.save_folder = os.path.normpath(if_str(save_folder, "save_folder"))

        self.dataset = if_str(dataset, "dataset").lower()

        if feature_type.upper() in self.__feature_types:
            self.feature_type = feature_type.upper()
        else:
            raise AttributeError(f"feature_type must be one of: {self.__feature_types}")

        if label_type.lower() in self.__label_types:
            self.label_type = label_type.lower()
        else:
            raise AttributeError(f"label_type must be one of: {self.__label_types}")

        self.repeated = if_bool(repeated, "repeated")
        self.energy = if_bool(energy, "energy")

        if (isinstance(deltas, (list, tuple))
                and len(deltas) == 2
                and isinstance(deltas[0], int)
                and isinstance(deltas[0], int)):
            self.deltas = deltas
        else:
            raise AttributeError(f"deltas must be length 2 tuple/list with int values inside it")

        self.nbanks = if_int(nbanks, "nbanks")
        self.filter_nan = if_bool(filter_nan, "filter_nan")
        self.sort = if_bool(sort, "sort")

        self.label_max_duration = if_float(label_max_duration, "label_max_duration")
        self.speeds = speeds if [if_float(s) for s in speeds] else self.__speeds

        self.debug = if_bool(debug, "debug")

        self.bigrams = True if label_type == self.__label_types[1] else False
        self.full_save_path = os.path.join(self.save_folder,
                                           f'{self.dataset.upper()}_{self.feature_type}_{self.label_type}'
                                           f'_{self.nbanks}_banks{"_DEBUG" if self.debug else ""}/')
        # 02_feature_length_range params
        self.min_frame_length = if_int(min_frame_length)
        self.max_frame_length = if_int(max_frame_length)
        self.mode = mode if if_str(mode) in self.__modes else self.__modes[0]
        self.delete_unused = if_bool(delete_unused)
        self.feature_names = if_str(feature_names)
        self.label_names = if_str(label_names)

        # 03_sort_data params
        self.tt_split_ratio = if_float(tt_split_ratio)  # TODO: range between 0. and 1.
        self.train_shard_size = train_shard_size
        self.test_shard_size = test_shard_size

        # 04_numpy_to_tfrecord
        self.delete_converted = if_bool(delete_converted)

        # for data_config.json
        self._num_features = None
        self._data_config_dict = dict()

    @staticmethod
    def _get_file_paths(audio_folder, transcript_folder):
        audio_files = [os.path.splitext(f) for f in os.listdir(audio_folder)
                       if os.path.isfile(os.path.join(audio_folder, f))]
        transcript_files = [os.path.splitext(f) for f in os.listdir(transcript_folder)
                            if os.path.isfile(os.path.join(transcript_folder, f))]

        files = []
        for file1, file2 in zip(audio_files, transcript_files):
            err_message = "{} =/= {}".format(file1[0], file2[0])
            assert file1[0] == file2[0], err_message
            files.append((f"{audio_folder}/{file1[0]}{file1[1]}", f"{transcript_folder}/{file2[0]}{file2[1]}"))

        return files

    @staticmethod
    def _get_file_names(files):
        return [os.path.splitext(os.path.split(file[0])[1])[0] for file in files]

    # 01_prepare_data.py
    def prepare_data(self, files):
        cepstra_length_list = []

        label_max_duration = self.label_max_duration
        speeds = self.speeds

        file_names = self._get_file_names(files)

        for speed in speeds:
            LOGGER.info(f"Create audio_transormer for speed {speed}")
            audio_transformer = (AudioEffectsChain().speed(speed))
            save_path = os.path.join(self.full_save_path, f"{speed}/")
            LOGGER.debug(f"Current save_path: {save_path}")
            for i, file in enumerate(files):
                if self.dataset == "pdtsc":
                    pdtsc = PDTSCLoader([file[0]], [file[1]], self.bigrams, self.repeated)
                    labels = pdtsc.transcripts_to_labels()  # list of lists of 1D numpy arrays
                    labels = labels[0]  # flatten label list
                    audio_list, fs = pdtsc.load_audio()
                    audio = audio_list[0]
                    fs = fs[0]
                    LOGGER.debug(
                        f"Loaded PDTSC with fs {fs} from:\n \t audio_path: {file[0]}\n \t transcript_path: {file[1]}")
                elif self.dataset == "oral":
                    oral = OralLoader([file[0]], [file[1]], self.bigrams, self.repeated)
                    label_dict = oral.transcripts_to_labels(
                        label_max_duration)  # Dict['file_name':Tuple[sents_list, starts_list, ends_list]]
                    audio_dict, fs_dict = oral.load_audio()  # Dicts['file_name']

                    labels = label_dict[file_names[i]]
                    audio = audio_dict[file_names[i]]
                    fs = fs_dict[file_names[i]]
                    LOGGER.debug(
                        f"Loaded ORAL with fs {fs} from:\n \t audio_path: {file[0]}\n \t transcript_path: {file[1]}")
                else:
                    raise ValueError("'dataset' argument must be either 'pdtsc' or 'oral'")

                full_save_path = os.path.join(save_path, file_names[i])

                LOGGER.info(f"\tApplying SoX transormation on audio from {full_save_path}")
                for ii in range(len(audio)):
                    LOGGER.debug(f"\t\t input.shape: {audio[ii].shape}")
                    audio[ii] = audio_transformer(audio[ii])
                    LOGGER.debug(f"\t\t output.shape: {audio[ii].shape}")

                LOGGER.info(f"\tApplying FeatureExtractor on audio")
                feat_ext = FeatureExtractor(audio, fs, feature_type=self.feature_type, energy=self.energy,
                                            deltas=self.deltas, nbanks=self.nbanks)
                cepstra = feat_ext.transform_data()  # list of 2D arrays
                # filter out cepstra which are containing nan values
                if self.filter_nan:
                    LOGGER.info(f"\tFiltering out NaN values")
                    # boolean list where False marks cepstra in which there is at least one nan value present
                    mask_nan = [not np.isnan(cepstrum).any() for cepstrum in cepstra]

                    # mask out cepstra and their corresponding labels with nan values
                    cepstra = list(compress(cepstra, mask_nan))
                    labels = list(compress(labels, mask_nan))

                # SAVE Cepstra to files (features)
                LOGGER.info(f"\tSaving cepstra to files")
                FeatureExtractor.save_cepstra(cepstra, full_save_path, exist_ok=True)
                LOGGER.debug(f"\t\tfull_save_path: {full_save_path}")

                # SAVE Transcripts to files (labels)
                LOGGER.info(f"\tSaving transcripts to files")
                if self.dataset == 'pdtsc':
                    pdtsc.save_labels([labels], save_path, exist_ok=True)
                elif self.dataset == 'oral':
                    label_dict[file_names[i]] = labels
                    oral.save_labels(label_dict, save_path, exist_ok=True)
                else:
                    raise ValueError("'dataset' argument must be either 'pdtsc' or 'oral'")

                LOGGER.info(f"\tChecking SAVE/LOAD consistency")
                loaded_cepstra, loaded_cepstra_paths = FeatureExtractor.load_cepstra(full_save_path)
                loaded_labels, loaded_label_paths = DataLoader.load_labels(full_save_path)

                # flatten the lists
                loaded_cepstra, loaded_cepstra_paths, loaded_labels, loaded_label_paths = (loaded_cepstra[0],
                                                                                           loaded_cepstra_paths[0],
                                                                                           loaded_labels[0],
                                                                                           loaded_label_paths[0])

                for j in range(len(cepstra)):
                    if np.any(np.not_equal(cepstra[j], loaded_cepstra[j])):
                        raise UserWarning("Saved and loaded cepstra are not value consistent.")
                    if self.dataset == 'pdtsc':
                        if np.any(np.not_equal(labels[j], loaded_labels[j])):
                            raise UserWarning("Saved and loaded labels are not value consistent.")
                    elif self.dataset == 'oral':
                        if np.any(np.not_equal(labels[j][0], loaded_labels[j])):
                            raise UserWarning("Saved and loaded labels are not value consistent.")

                    # add (cepstrum_path, label_path, cepstrum_length) tuple into collective list for sorting
                    cepstra_length_list.append(
                        (loaded_cepstra_paths[j], loaded_label_paths[j], loaded_cepstra[j].shape[0]))
                LOGGER.debug(f'files from {file_names[i]} transformed and saved into {os.path.abspath(save_path)}.')

            # sort cepstra and labels by time length (number of frames)
            if self.sort:
                LOGGER.info(f"Sorting cepstra and labels by time length (number of frames)")
                sort_indices = np.argsort(
                    [c[2] for c in cepstra_length_list])  # indices which sort the lists by cepstra length
                cepstra_length_list = [cepstra_length_list[i] for i in sort_indices]  # sort the cepstra list

                num_digits = len(str(len(cepstra_length_list)))

                for idx, file in enumerate(cepstra_length_list):
                    cepstrum_path, label_path, _ = file
                    os.rename(cepstrum_path, "{0}/cepstrum-{1:0{2}d}.npy".format(save_path, idx, num_digits))
                    os.rename(label_path, "{0}/transcript-{1:0{2}d}.npy".format(save_path, idx, num_digits))
                subfolders = next(os.walk(save_path))[1]
                for folder in subfolders:
                    try:
                        os.rmdir(os.path.join(save_path, folder))
                    except OSError:
                        LOGGER.warning("Folder {} is not empty! Can't delete.".format(os.path.join(save_path, folder)))
        LOGGER.info(f"Save the number of features in cepstra.")
        self._num_features = cepstra[0].shape[1]
        LOGGER.debug(f"_num_features: {self._num_features}")

    # 02_feature_length_range.py
    def feature_length_range(self):
        """ Check individual files (features and their labels) in load_dir and copy/move those which satisfy the condition:
        min_frame_length <= feature_frame_len <= max_frame_length

        :param load_dir: folder from which to load features and their labels
        :param min_frame_length: lower bound of the feature frame length condition
        :param max_frame_length: upper bound of the feature frame length condition
        :param mode: 'copy'/'move' - condition satisfying files are copied/moved from load_dir to save_dir
        :param feature_names: sequence of symbols that can be used as common identifier for feature files
        :param label_names: sequence of symbols that can be used as common identifier for label files
        :return: None
        """

        # normalize the save directory path
        save_path = f"{self.full_save_path[:-1]}_min_{self.min_frame_length}_max_{self.max_frame_length}/"

        folder_structure_gen = os.walk(self.full_save_path)  # ('path_to_current_folder', [subfolders], ['files', ...])

        for folder in folder_structure_gen:
            path, subfolders, files = folder
            feat_file_names = [f for f in files if self.feature_names in f]
            label_file_names = [f for f in files if self.label_names in f]

            num_feats = len(feat_file_names)
            num_labels = len(label_file_names)

            assert num_feats == num_labels, 'There is {} feature files and {} label files (must be same).'.format(
                num_feats, num_labels)

            rel_path = os.path.relpath(path,
                                       self.full_save_path)  # relative position of current subdirectory in regards to load_dir
            save_full_path = os.path.join(save_path, rel_path)  # folder/subfolder to which save files in save_dir

            # make subdirectories in save_dir
            os.makedirs(save_full_path, exist_ok=True)

            for i in range(num_feats):
                feat_load_path = os.path.join(path, feat_file_names[i])
                label_load_path = os.path.join(path, label_file_names[i])
                feat_save_path = os.path.join(save_full_path, feat_file_names[i])
                label_save_path = os.path.join(save_full_path, label_file_names[i])

                feat, _ = FeatureExtractor.load_cepstra(feat_load_path)
                feat_frame_len = feat[0][0].shape[0]

                if self.min_frame_length <= feat_frame_len <= self.max_frame_length:
                    if self.mode == 'copy':
                        shutil.copy2(feat_load_path, feat_save_path)
                        LOGGER.debug("Copied {} to {}".format(feat_load_path, feat_save_path))
                        shutil.copy2(label_load_path, label_save_path)
                        LOGGER.debug("Copied {} to {}".format(label_load_path, label_save_path))
                    elif self.mode == 'move':
                        os.rename(feat_load_path, feat_save_path)
                        LOGGER.debug("Moved {} to {}".format(feat_load_path, feat_save_path))
                        os.rename(label_load_path, label_save_path)
                        LOGGER.debug("Moved {} to {}".format(label_load_path, label_save_path))
                    else:
                        raise ValueError("argument mode must be either 'copy' or 'move'")

        # Delete remaining unmoved files
        if self.delete_unused:
            LOGGER.info(f"Removing remaining files and folder at path: {self.full_save_path}.")
            shutil.rmtree(self.full_save_path, ignore_errors=True)

        self.full_save_path = save_path
        LOGGER.info(f"Full save path changed to: {self.full_save_path}")
        LOGGER.info("Finished.")

    # 03_sort_data.py
    def _get_sorted_lists_by_file_size(self, save_path):

        path_gen = os.walk(save_path)

        file_size_list = []
        sorted_train_list = []
        sorted_test_list = []

        for path, subfolders, files in path_gen:
            fullpaths = []
            for file in files:
                if self.feature_names in file:
                    label_fullpath = os.path.join(path, file.replace(self.feature_names,
                                                                     self.label_names))  # corresponding label path
                    if os.path.exists(label_fullpath):
                        fullpaths.append((os.path.join(path, file), label_fullpath))
                    else:
                        message = 'corresponding label file not found at path {}'.format(label_fullpath)
                        raise FileNotFoundError(message)

            file_size_list.extend(zip(fullpaths, [os.path.getsize(fp[0]) for fp in fullpaths]))

        sorted_file_size_list = sorted(file_size_list, key=lambda x: x[1])

        LOGGER.info("Splitting into train and test datasets by tt_split_ratio")
        data_len = len(sorted_file_size_list)
        test_period = 1.0//(1.0 - self.tt_split_ratio)

        for i in range(data_len):
            if (i + 1) % test_period:
                # put into train_dataset
                sorted_train_list.append(sorted_file_size_list[i])
            else:
                # put into test_dataset
                sorted_test_list.append(sorted_file_size_list[i])
        LOGGER.info("File-size lists for train and test datasets sorted and saved.")

        return sorted_train_list, sorted_test_list

    def move_to_shard_folders(self, sorted_list, shard_size=1024, speed=1.0, subset=None):
        """

        :param sorted_list:
        :param shard_size: approximate size of folder shards in which the data is split in MBytes
        :param speed: speed parameter of SpeedAug
        :param subset: either "train" or "test"
        :return: None
        """
        # data_config dictionary initialization for current speed
        if not str(speed) in self._data_config_dict.keys():
            LOGGER.info(f"Initializing _data_config_dict[{speed}]")
            self._data_config_dict[str(speed)] = {"num_train_data": 0,
                                                  "num_test_data": 0,
                                                  "num_rest_data": 0,
                                                  "num_features": self._num_features,
                                                  "min_time": self.min_frame_length,
                                                  "max_time": self.max_frame_length}

        data_len = len(sorted_list)
        data_size = sum(sfs[1] for sfs in sorted_list)
        subfolder = f"{speed}/{subset}"

        byte_min_shard_size = shard_size * 1e6  # convert to Byte size
        max_num_shards = int(data_size // byte_min_shard_size + 1)
        num_data_digits = len(str(data_len))
        num_shard_digits = len(str(max_num_shards))

        current_shard_size = 0
        file_idx = 0
        shard_idx = 0

        save_folder = os.path.join(self.full_save_path, subfolder)
        os.makedirs(save_folder, exist_ok=True)
        LOGGER.info('Save folder set to {}'.format(save_folder))

        for i, ((feature_path, label_path), size) in enumerate(sorted_list):
            current_shard_folder = os.path.join(save_folder, 'shard_{0:0{1}d}'.format(shard_idx, num_shard_digits))
            os.makedirs(current_shard_folder, exist_ok=True)
            new_feature_name = '{0}-{1:0{2}d}.npy'.format(self.feature_names, file_idx, num_data_digits)
            new_label_name = '{0}-{1:0{2}d}.npy'.format(self.label_names, file_idx, num_data_digits)
            os.rename(feature_path, os.path.join(current_shard_folder, new_feature_name))
            os.rename(label_path, os.path.join(current_shard_folder, new_label_name))
            if "train" in subset:
                self._data_config_dict[str(speed)]["num_train_data"] += 1
            elif "test" in subset:
                self._data_config_dict[str(speed)]["num_test_data"] += 1
            else:
                self._data_config_dict[str(speed)]["num_rest_data"] += 1
            if current_shard_size < byte_min_shard_size:
                file_idx += 1
                current_shard_size += size
            else:
                LOGGER.info('Files for shard number {} saved to folder {}'.format(shard_idx, current_shard_folder))
                file_idx = 0
                shard_idx += 1
                current_shard_size = 0
        LOGGER.info(f"{subfolder} subfolder filled with sorted shards.")
        LOGGER.debug(f"_data_config_dict[{speed}]: {self._data_config_dict[str(speed)]}")

    # 04_numpy_to_tfrecord.py
    @staticmethod
    def _serialize_array(x, y):
        feature = {
            'x': tf.train.Feature(float_list=tf.train.FloatList(value=x.flatten())),
            'y': tf.train.Feature(int64_list=tf.train.Int64List(value=y.flatten()))
        }
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        return example.SerializeToString()

    def numpy_to_tfrecord(self):

        input_folder = os.path.normpath(self.full_save_path)
        output_folder = input_folder + "_tfrecord"
        folder_structure_gen = os.walk(input_folder)  # ('path_to_current_folder', [subfolders], ['files', ...])

        for folder in folder_structure_gen:
            path, subfolders, files = folder
            if not files:
                continue
            feat_file_names = [f for f in files if self.feature_names in f]
            label_file_names = [f for f in files if self.label_names in f]

            if output_folder and isinstance(output_folder, str):
                output_path = os.path.join(os.path.normpath(output_folder), *path.split("\\")[-3:])
                os.makedirs(os.path.split(output_path)[0], exist_ok=True)
            else:
                output_path = os.path.splitext(path)[0]

            num_feats = len(feat_file_names)
            num_labels = len(label_file_names)

            assert num_feats == num_labels, 'There is {} feature files and {} label files (must be same).'.format(
                num_feats,
                num_labels)

            tfrecord_path = output_path + '.tfrecord'
            writer = tf.io.TFRecordWriter(tfrecord_path)

            for i in range(num_feats):
                feat_load_path = os.path.join(path, feat_file_names[i])
                label_load_path = os.path.join(path, label_file_names[i])

                feat, _ = FeatureExtractor.load_cepstra(feat_load_path)
                label, _ = DataLoader.load_labels(label_load_path)

                #            print(feat[0][0].shape, label[0][0].shape)

                serialized = self._serialize_array(feat[0][0], label[0][0])
                writer.write(serialized)

            writer.close()

            if self.delete_converted:
                LOGGER.info(f"Removing shard folder with .npy files at path: {path}")
                shutil.rmtree(path, ignore_errors=True)

            LOGGER.debug("Data written to {}".format(tfrecord_path))
        LOGGER.info(f"All shards converted to tfrecords and saved to folder {output_folder}")

        if self.delete_converted:
            LOGGER.info(f"Removing empty folders.")
            for root, _, files in os.walk(self.full_save_path, topdown=False):
                try:
                    os.rmdir(root)
                    LOGGER.debug(f"Path {root} DELETED (was empty).")
                except OSError:
                    LOGGER.warning(f"Path {root} NOT DELETED (was not empty).")

    # 05_save_data_config.json
    def save_data_config(self):
        for key, config_dict in self._data_config_dict:
            data_config_path = f"{self.full_save_path[:-1]}_tfrecord/{key}/data_config.json"
            with open(data_config_path, "w") as f:
                LOGGER.info(f"Saving data_config to path: {data_config_path}")
                json.dump(config_dict, f)

    def run(self):
        LOGGER.info("01_prepare_data")
        files = self._get_file_paths(self.audio_folder, self.transcript_folder)
        self.prepare_data(files)

        LOGGER.info("02_feature_length_range")
        self.feature_length_range()

        LOGGER.info("03_sort_data")
        for speed in self.speeds:
            save_folder = os.path.join(self.full_save_path, str(speed))
            sorted_train_list, sorted_test_list = self._get_sorted_lists_by_file_size(save_folder)
            self.move_to_shard_folders(sorted_train_list, self.train_shard_size, speed, "train")
            self.move_to_shard_folders(sorted_test_list, self.test_shard_size, speed, "test")

        LOGGER.info("04_numpy_to_tfrecord")
        self.numpy_to_tfrecord()

        LOGGER.info("05_save_data_config")
        self.save_data_config()

        LOGGER.info("DataPrep finished!")
