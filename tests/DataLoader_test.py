import os

from DataOps import PDTSCLoader

if __name__ == '__main__':

    audio_folder = "c:/!temp/raw_debug/audio"
    transcript_folder = "c:/!temp/raw_debug/transcripts"
    audio_files = [os.path.join(audio_folder, f) for f in os.listdir(audio_folder)
                   if os.path.isfile(os.path.join(audio_folder, f))]
    transcript_files = [os.path.join(transcript_folder, f) for f in os.listdir(transcript_folder)
                        if os.path.isfile(os.path.join(transcript_folder, f))]

    pdtsc = PDTSCLoader(audio_files, transcript_files, bigrams=False, repeated=False)
    pdtsc.transcripts_to_labels()
    print(pdtsc.tokens[0][0])
    print(pdtsc.labels[0][0])
    print(pdtsc.load_audio())
#    pdtsc.save_audio('./data/test_saved.ogg', pdtsc.audio[0][1], pdtsc.fs[0])
#    pdtsc.save_labels(folder='./data/train', exist_ok=True)