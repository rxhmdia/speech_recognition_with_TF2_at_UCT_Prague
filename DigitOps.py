# Transcribing some czech numbers to their digit version
import re

from collections import defaultdict


def _combine_digits(d2re_base):
    d2re = defaultdict(lambda: r"")
    for d in range(0, 91, 10):  # 0, 10, 20, 30, ... 90
        for j in range(0, 10):  # 0, 1, 2, ... 9
            key = d + j
            if key in d2re_base.keys():
                d2re[key] = d2re_base[key]
            else:
                d2re[key] = r" ".join([d2re_base[d], d2re_base[j]])
    return d2re


class DigitTranscriber:
    D2RE_BASE = {0: r"\bnul\w+",
                 1: r"(\bjedn\w+|\bprvní|\bprvý|\bprvej)",
                 2: r"(\bdva|\bdvě|\bdvoj\w+|\bdruh\w+)",
                 3: r"(\btři|\btroj\w+|\btřetí)",
                 4: r"(\bčtyř\w*|čtvrt\w+|\bštyř\w*)",
                 5: r"(\bpět|\bpětk\w+|\bpáté\w+)",
                 6: r"\bšest\w*",
                 7: r"(\bsedm\w*|\bsedum)",
                 8: r"(\bosm\w*|\bosum)",
                 9: r"(\bdevět|\bdevát\w+|\bdevít\w+)",
                 10: r"(\bdeset|\bdesát\w+|\bdesít\w+)",
                 11: r"\bjedenáct\w*",
                 12: r"\bdvanáct\w*",
                 13: r"\btřináct\w*",
                 14: r"\bčtrnáct\w*",
                 15: r"\bpatnáct\w*",
                 16: r"\bšestnáct\w*",
                 17: r"\bsedu?mnáct\w*",
                 18: r"\bosu?mnáct\w*",
                 19: r"\bdevatenáct\w*",
                 20: r"(\bdvacet|\bdvacát\w+|\bdvacít\w+)",
                 30: r"(\btřicet|\btřicát\w+|\btřicít\w+)",
                 40: r"(\b[šč]tyřicet|\b[šč]tyřicát\w+|\b[šč]tyřicít\w+)",
                 50: r"\bpadesát\w*",
                 60: r"\bšedesát\w*",
                 70: r"(\bsedmdesát\w*|\bsedumdesát\w*)",
                 80: r"(\bosmdesát\w*|\bosumdesát\w*)",
                 90: r"\bdevadesát\w*"}

    d2re = _combine_digits(D2RE_BASE)
    d2re_list = [(k, v) for k, v in d2re.items()]
    d2re_list.sort(key=lambda tup: tup[0], reverse=True)  # sort from highest to lowest key
    d2re_tuple = tuple(d2re_list)

    def __init__(self):
        self._counter = defaultdict(lambda: 0)

    def transcribe(self, sent):
        for k, v in self.d2re_tuple:
            sent, count = re.subn(v, str(k), sent, flags=re.I)
            self._counter[k] += count
        return sent

    def get_counter(self):
        return self._counter

    def count_nonzero(self):
        return {k: v for k, v in self._counter.items() if v > 0}

    def print_nonzero_counts(self):
        print(self.count_nonzero().items())


if __name__ == '__main__':
    dt = DigitTranscriber()

    sentence = "To jsem takhle poprvé první den jednou v osumnáct osmnáct a sedmnáct sedumnáct a sedum nebo osm minut jel dvacet devítkou a pak vystoupil na osmičce a došel za patnáct hodin do Berouna a pak štyřicet štyři hodin pěšky."

    transcribed_sentence = dt.transcribe(sentence)

    print(transcribed_sentence)

    print(dt.count_nonzero())

