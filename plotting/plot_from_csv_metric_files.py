import os
import pandas as pd
from matplotlib import pyplot as plt


def csv_to_df_dict(folder, metric_names=('mean_cer', 'mean_loss')):
    path_gen = os.walk(os.path.normpath(folder))

    df = dict()
    for folder_path, subfolders, files in path_gen:
        folder_name = os.path.split(folder_path)[-1]
        fullpaths = [os.path.join(folder_path, file) for file in files if '.csv' in file]
        df[folder_name] = dict()
        # TODO: load same metric files into lists
        for metric_name in metric_names:
            # load metric specific csv files into df dictionary
            for path in fullpaths:
                if metric_name in path:
                    if metric_name not in df[folder_name].keys():
                        df[folder_name][metric_name] = pd.read_csv(path)
                    else:
                        df_shard = pd.read_csv(path)
                        df[folder_name][metric_name] = pd.concat([df[folder_name][metric_name],
                                                                  df_shard], ignore_index=True)
            # rename Value columns to corresponding metric name
            if metric_name in df[folder_name].keys():
                df[folder_name][metric_name].rename(columns={'Value': metric_name},
                                                    inplace=True)

    return df


def plot_and_save_dataframes(df_dict, metric_label_dict, save_folder='results'):
    """
        :param df_dict: dictionary of dictionaries of dataframes (df_dict[folder][metric])
        :param metric_label_dict: mapping of metric_names to title, xlabel and ylabel
        :param save_folder: folder to which the plots will be saved
    """
    for folder, metric_dict in df_dict.items():
        for metric_name, dataframe in metric_dict.items():
            plt.figure()
            plot = dataframe.plot(y=-1,
                                  title=metric_label_dict[metric_name][0],
                                  grid=True)
            plot.set_xlabel(metric_label_dict[metric_name][1])
            plot.set_ylabel(metric_label_dict[metric_name][2])

            full_save_path = os.path.join(os.path.normpath(save_folder),
                                          folder, metric_name + '.pdf')

            os.makedirs(os.path.split(full_save_path)[0], exist_ok=True)

            plt.savefig(full_save_path, transparent=True)


# TODO: plot from multiple files
# TODO: calculate mean and variance of the minima


if __name__ == '__main__':
    folder = 'd:/!private/lord/git/speech_recognition/data/results/06_ff_at_end/MFSC/_cer[0.33]_conv[64-128-256](bn)_rnn[512-512](bn)_ff[256-128](bn)/plots/'
    metric_names = ['mean_cer', 'total_loss']
    metric_label_dict = {metric_names[0]: ('Character Error Rate (CER)', 'epoch', 'CER'),
                         metric_names[1]: ('CTC loss', 'epoch', 'loss')}

    df = csv_to_df_dict(folder, metric_names=metric_names)

    plot_and_save_dataframes(df, metric_label_dict, save_folder=folder)