from pathlib import Path

import pandas as pd

from DigitOps import DigitTranscriber

FOLDER_WITH_TRANSCRIPTS = Path("b:/!DATASETS/CommonVoice/cs/")
PATHS_TO_TRANSCRIPTS = [str(f) for f in FOLDER_WITH_TRANSCRIPTS.glob("**/*.tsv")]

if __name__ == '__main__':
    print(PATHS_TO_TRANSCRIPTS)

    dt = DigitTranscriber()

    for pth in PATHS_TO_TRANSCRIPTS:
        df = pd.read_csv(pth, sep="\t")
        for i in range(len(df)):
            # sent = bytes(df.iloc[i, 2], encoding="utf8").decode("utf8")
            sent = df.iloc[i, 2]
            df.iloc[i, 2] = dt.transcribe(sent)

        *pth_to_fl, fl = pth.split("\\")
        flnm, flext = fl.split(".")
        pth_to_fl = "\\".join(pth_to_fl)+"\\"

        with open(pth_to_fl+flnm+"_d."+flext, "w", encoding="utf8") as f:
            df.to_csv(f, sep="\t", index=False)

    print(dt.counter)


