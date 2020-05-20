from DataOps import DataPrep

if __name__ == '__main__':
    audio_folder = "b:/!DATASETS/oral2013/audio/"
    transcript_folder = "b:/!DATASETS/oral2013/transcripts/"
    save_folder = 'B:/!temp/'

    dp = DataPrep(audio_folder,
                  transcript_folder,
                  save_folder,
                  dataset="oral",
                  speeds=(1.0, ),
                  mode="move",
                  tt_split_ratio=1.0,
                  delete_unused=True,
                  delete_converted=True,
                  debug=False)

    dp.run()