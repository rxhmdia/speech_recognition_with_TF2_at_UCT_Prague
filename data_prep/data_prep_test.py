from DataOps import DataPrep

if __name__ == '__main__':
    audio_folder = "b:/!DATASETS/PDTSC/audio/"
    transcript_folder = "b:/!DATASETS/PDTSC/transcripts/"
    save_folder = 'B:/!temp/'

    dp = DataPrep(audio_folder,
                  transcript_folder,
                  save_folder,
                  speeds=(1.0, ),
                  mode="move",
                  delete_unused=False,
                  delete_converted=False,
                  debug=False)

    dp.run()