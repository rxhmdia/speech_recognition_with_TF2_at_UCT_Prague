from DataOps import DataPrep

if __name__ == '__main__':
    audio_folder = "b:/!DATASETS/raw_debug/audio/"
    transcript_folder = "b:/!DATASETS/raw_debug/transcripts/"
    save_folder = 'B:/!temp/'

    dp = DataPrep(audio_folder,
                  transcript_folder,
                  save_folder,
                  speeds=(0.9, 1.0),
                  train_shard_size=10,
                  mode="move",
                  debug=True)

    dp.run()