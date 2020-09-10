# Transcribing some czech numbers to their digit version
import re

from collections import defaultdict


class DigitTranscriber:
    D2RE_BASE = {0: r"\bnul\w+",
                 1: r"\bjedn\w+",
                 2: r"(\bdva|\bdvě|\bdvoj\w+|\bdruh\w+)",
                 3: r"(\btři|\btroj\w+|\btřetí)",
                 4: r"(\bčtyři|čtvrt\w+|\bčtyřk\w+)",
                 5: r"(\bpět|\bpětk\w+|\bpáté\w+)",
                 6: r"\bšest\w*",
                 7: r"\bsedm\w*",
                 8: r"\bosm\w*",
                 9: r"(\bdevět|\bdevát\w+|\bdevít\w+)",
                 10: r"(\bdeset|\bdesát\w+|\bdesít\w+)",
                 11: r"\bjedenáct\w*",
                 12: r"\bdvanáct\w*",
                 13: r"\btřináct\w*",
                 14: r"\bčtrnáct\w*",
                 15: r"\bpatnáct\w*",
                 16: r"\bšestnáct\w*",
                 17: r"\bsedmnáct\w*",
                 18: r"\bosmnáct\w*",
                 19: r"\bdevatenáct\w*",
                 20: r"(\bdvacet|\bdvacát\w+|\bdvacít\w+)",
                 30: r"(\btřicet|\btřicát\w+|\btřicít\w+)",
                 40: r"(\bčtyřicet|\bčtyřicát\w+|\bčtyřicít\w+)",
                 50: r"\bpadesát\w*",
                 60: r"\bšedesát\w*",
                 70: r"\bsedmdesát\w*",
                 80: r"\bosmdesát\w*",
                 90: r"\bdevadesát\w*"}

    def __init__(self):
        self.d2re = self._combine_digits()

        d2re_list = [(k, v) for k, v in self.d2re.items()]
        d2re_list.reverse()  # higher numbers first
        self.d2re_tuple = tuple(d2re_list)

        self._counter = defaultdict(lambda: 0)

    def transcribe(self, sentence):
        for k, v in self.d2re_tuple:
            sentence, count = re.subn(v, str(k), sentence, flags=re.I)
            self._counter[k] += count
        return sentence

    def get_counter(self):
        return self._counter

    def count_nonzero(self):
        return {k: v for k, v in self._counter.items() if v > 0}

    def print_nonzero_counts(self):
        print(self.count_nonzero().items())

    def _combine_digits(self):
        d2re_base = self.D2RE_BASE
        d2re = defaultdict(lambda: r"")
        for d in range(0, 91, 10):  # 0, 10, 20, 30, ... 90
            for j in range(0, 10):  # 0, 1, 2, ... 9
                key = d + j
                if key in d2re_base.keys():
                    d2re[key] = d2re_base[key]
                else:
                    d2re[key] = r" ".join([d2re_base[d], d2re_base[j]])
        return d2re

if __name__ == '__main__':
    dt = DigitTranscriber()

    sentence = "To jsem takhle jel dvacet devítkou a pak vystoupil na osmičce a došel za patnáct hodin do Berouna."

    transcribed_sentence = dt.transcribe(sentence)

    print(transcribed_sentence)

    print(dt.counter)

