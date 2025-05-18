# Third Party
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from sklearn.neighbors import KernelDensity

font = 'DejaVu Sans'

dark_red = '#4F1748'
red = '#9D2F92'

dark_blue = '#19779E'
blue = '#199FD3'

dark_pa_blue = '#075cab'
pa_blue = '#007AFF'

turquoise = '#26f8e6'
dark_turquoise = '#0e8a7f'

royal_blue = '#27beff'
dark_royal_blue = '#13699e'

red_pink = '#e5499c'
dark_red_pink = '#8a0f50'

purple = '#b361e5'
dark_purple = '#68149c'

title_size = 22  # 22 otherwise, 18 ecg
subtitle_size = 20
axes_size = 18
legend_size = 14


def epochs():
    FOLDER_NAME = "Epoch Tests"
    RESULTS = "epoch_results.csv"

    # For plotting specific results
    SPECIFIC_NAME_1 = "TVAE_07-05-24-1244_1000_1000_5000_25_kde_mia_domias_evaluation_scott"  # Of specific graph
    SPECIFIC_NAME_2 = "TVAE_07-05-24-1432_1000_1000_5000_1500_kde_mia_domias_evaluation_scott"  # Of specific graph

    all_results = pd.read_csv(f'Saved Data/{FOLDER_NAME}/{RESULTS}').drop(
        ['Generator', 'Date', 'MIA Method', 'Bandwidth'], axis=1).to_numpy(dtype=np.float32)

    mem_set = pd.read_csv(f'Saved Data/{FOLDER_NAME}/{SPECIFIC_NAME_1}/mem_set.csv').drop(['Unnamed: 0'],
                                                                                          axis=1).to_numpy(
        dtype=np.float32)
    reference_set = pd.read_csv(f'Saved Data/{FOLDER_NAME}/{SPECIFIC_NAME_1}/ref_set.csv').drop(['Unnamed: 0'],
                                                                                                axis=1).to_numpy(
        dtype=np.float32)
    synth_set = pd.read_csv(f'Saved Data/{FOLDER_NAME}/{SPECIFIC_NAME_1}/synth_set.csv').drop(['Unnamed: 0'],
                                                                                              axis=1).to_numpy(
        dtype=np.float32)
    x_test = pd.read_csv(f'Saved Data/{FOLDER_NAME}/{SPECIFIC_NAME_1}/x_test.csv').drop(['Unnamed: 0'],
                                                                                        axis=1).to_numpy(
        dtype=np.float32)
    y_test = pd.read_csv(f'Saved Data/{FOLDER_NAME}/{SPECIFIC_NAME_1}/y_test.csv').drop(['Unnamed: 0'],
                                                                                        axis=1).to_numpy(
        dtype=np.bool_).flatten()
    y_pred = pd.read_csv(f'Saved Data/{FOLDER_NAME}/{SPECIFIC_NAME_1}/y_pred.csv').drop(['Unnamed: 0'],
                                                                                        axis=1).to_numpy(
        dtype=np.bool_).flatten()

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot Top
    plt.subplot(2, 1, 1)  # divide as 2x1, plot top
    plt.plot(all_results[:, 3], all_results[:, 5], color=dark_royal_blue)  # AUC test 1

    all_results2 = pd.read_csv(f'Saved Data/Epoch Tests 2/epoch_results2.csv').drop(
        ['Generator', 'Date', 'MIA Method', 'Bandwidth'], axis=1).to_numpy(dtype=np.float32)
    error = abs(all_results2[:, 5] - all_results[:, 5]) * all_results[:, 5]
    y_error_pos = all_results[:, 5] + error
    y_error_neg = all_results[:, 5] - error
    plt.fill_between(all_results[:, 3], all_results[:, 5] - error, all_results[:, 5] + error, color=royal_blue,
                     alpha=0.5, edgecolor='#FFFFFF')
    plt.axhline(y=0.5, color='k', linestyle='--')
    plt.xlim([5, 10000])
    plt.ylim([0.48, 0.72])
    plt.title('MIA AUC Against Number of Epochs',
              fontname=font,
              fontsize=title_size)
    plt.xlabel('Number of Epochs', fontname=font, fontsize=axes_size)
    plt.ylabel('MIA AUC', fontname=font, fontsize=axes_size)
    plt.xscale('log')
    plt.xticks(fontname=font, fontsize=axes_size)
    plt.yticks(fontname=font, fontsize=axes_size)
    plt.legend(labels=['AUC', 'Percentage Error', 'Baseline Accuracy'],
               fontsize=legend_size,
               loc='upper left')

    plt.subplot(2, 2, 3)
    avg_ref_set = np.mean(reference_set, axis=0)
    avg_synth_set = np.mean(synth_set, axis=0)

    # Plotting the first x_set ECG KDE against the averages to show which it may be closer to, along with printing
    # its classification. Also plotting the last x_test (not from training) and assignment to see if it is closer to ref
    sns.kdeplot(avg_synth_set, fill=True, color=royal_blue)
    sns.kdeplot(x_test[26], color=dark_royal_blue)  # CHANGE
    sns.kdeplot(avg_ref_set, fill=True, color=red)
    sns.kdeplot(x_test[len(x_test) - 1], color=dark_red)
    plt.xlim([0, 1.2])

    plt.xticks(fontname=font, fontsize=axes_size)
    plt.yticks(fontname=font, fontsize=axes_size)
    plt.title('Densities of ECG Amplitudes for 25 Epochs',
              # for Prediction Success Comparison - changed from suptitle
              fontname=font,
              fontsize=title_size)

    plt.xlabel('ECG Amplitude', fontname=font, fontsize=axes_size)
    plt.ylabel('Density', fontname=font, fontsize=axes_size)
    plt.legend(labels=['Synthetic Data',
                       f'Training Sample | Prediction = {y_pred[26]}',
                       'Reference Data',
                       f'Reference Sample | Prediction = {y_pred[len(y_pred) - 1]}'],
               fontsize=legend_size)

    mem_set = pd.read_csv(f'Saved Data/{FOLDER_NAME}/{SPECIFIC_NAME_2}/mem_set.csv').drop(['Unnamed: 0'],
                                                                                          axis=1).to_numpy(
        dtype=np.float32)
    reference_set = pd.read_csv(f'Saved Data/{FOLDER_NAME}/{SPECIFIC_NAME_2}/ref_set.csv').drop(['Unnamed: 0'],
                                                                                                axis=1).to_numpy(
        dtype=np.float32)
    synth_set = pd.read_csv(f'Saved Data/{FOLDER_NAME}/{SPECIFIC_NAME_2}/synth_set.csv').drop(['Unnamed: 0'],
                                                                                              axis=1).to_numpy(
        dtype=np.float32)
    x_test = pd.read_csv(f'Saved Data/{FOLDER_NAME}/{SPECIFIC_NAME_2}/x_test.csv').drop(['Unnamed: 0'],
                                                                                        axis=1).to_numpy(
        dtype=np.float32)
    y_test = pd.read_csv(f'Saved Data/{FOLDER_NAME}/{SPECIFIC_NAME_2}/y_test.csv').drop(['Unnamed: 0'],
                                                                                        axis=1).to_numpy(
        dtype=np.bool_).flatten()
    y_pred = pd.read_csv(f'Saved Data/{FOLDER_NAME}/{SPECIFIC_NAME_2}/y_pred.csv').drop(['Unnamed: 0'],
                                                                                        axis=1).to_numpy(
        dtype=np.bool_).flatten()

    plt.subplot(2, 2, 4)  # divide as 2x2, plot top right
    # plt.plt(...)
    avg_ref_set = np.mean(reference_set, axis=0)
    avg_synth_set = np.mean(synth_set, axis=0)

    # Plotting the first x_set ECG KDE against the averages to show which it may be closer to, along with printing
    # its classification. Also plotting the last x_test (not from training) and assignment to see if it is closer to ref
    sns.kdeplot(avg_synth_set, fill=True, color=royal_blue)
    sns.kdeplot(x_test[0], color=dark_royal_blue)  # CHANGE
    sns.kdeplot(avg_ref_set, fill=True, color=red)
    sns.kdeplot(x_test[len(x_test) - 1], color=dark_red)
    plt.xlim([0, 1.2])

    plt.xticks(fontname=font, fontsize=axes_size)
    plt.yticks(fontname=font, fontsize=axes_size)
    plt.title('Densities of ECG Amplitudes for 1,500 Epochs',
              # for Prediction Success Comparison - changed from suptitle
              fontname=font,
              fontsize=title_size)

    plt.xlabel('ECG Amplitude', fontname=font, fontsize=axes_size)
    plt.ylabel('Density', fontname=font, fontsize=axes_size)
    plt.legend(labels=['Synthetic Data',
                       f'Training Sample | Prediction = {y_pred[0]}',
                       'Reference Data',
                       f'Reference Sample | Prediction = {y_pred[len(y_pred) - 1]}'],
               fontsize=legend_size)
    plt.tight_layout()
    plt.show()


def training():
    FOLDER_NAME = "train 1000, ref 1000, synth 5000, epochs 25/Training Set Tests"
    RESULTS = "training_results1.csv"

    # For plotting specific results
    SPECIFIC_NAME_1 = "TVAE_16-05-24-0921_500_1000_5000_25_kde_mia_domias_evaluation_scott"  # Of specific graph
    SPECIFIC_NAME_2 = "TVAE_16-05-24-0921_20000_1000_5000_25_kde_mia_domias_evaluation_scott"  # Of specific graph

    all_results = pd.read_csv(f'Saved Data/{FOLDER_NAME}/{RESULTS}').drop(
        ['Generator', 'Date', 'MIA Method', 'Bandwidth'], axis=1).to_numpy(dtype=np.float32)

    mem_set = pd.read_csv(f'Saved Data/{FOLDER_NAME}/{SPECIFIC_NAME_1}/mem_set.csv').drop(['Unnamed: 0'],
                                                                                          axis=1).to_numpy(
        dtype=np.float32)
    reference_set = pd.read_csv(f'Saved Data/{FOLDER_NAME}/{SPECIFIC_NAME_1}/ref_set.csv').drop(['Unnamed: 0'],
                                                                                                axis=1).to_numpy(
        dtype=np.float32)
    synth_set = pd.read_csv(f'Saved Data/{FOLDER_NAME}/{SPECIFIC_NAME_1}/synth_set.csv').drop(['Unnamed: 0'],
                                                                                              axis=1).to_numpy(
        dtype=np.float32)
    x_test = pd.read_csv(f'Saved Data/{FOLDER_NAME}/{SPECIFIC_NAME_1}/x_test.csv').drop(['Unnamed: 0'],
                                                                                        axis=1).to_numpy(
        dtype=np.float32)
    y_test = pd.read_csv(f'Saved Data/{FOLDER_NAME}/{SPECIFIC_NAME_1}/y_test.csv').drop(['Unnamed: 0'],
                                                                                        axis=1).to_numpy(
        dtype=np.bool_).flatten()
    y_pred = pd.read_csv(f'Saved Data/{FOLDER_NAME}/{SPECIFIC_NAME_1}/y_pred.csv').drop(['Unnamed: 0'],
                                                                                        axis=1).to_numpy(
        dtype=np.bool_).flatten()

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot Top
    plt.subplot(2, 1, 1)  # divide as 2x1, plot top
    plt.plot(all_results[:, 0], all_results[:, 5], color=dark_royal_blue)  # AUC
    all_results2 = pd.read_csv(
        f'Saved Data/train 1000, ref 1000, synth 5000, epochs 25/Training Set Tests 2/training_results2.csv').drop(
        ['Generator', 'Date', 'MIA Method', 'Bandwidth'], axis=1).to_numpy(dtype=np.float32)
    error = abs(all_results2[:, 5] - all_results[:, 5]) * all_results[:, 5]
    y_error_pos = all_results[:, 5] + error
    y_error_neg = all_results[:, 5] - error
    plt.fill_between(all_results[:, 0], all_results[:, 5] - error, all_results[:, 5] + error, color=royal_blue,
                     alpha=0.5, edgecolor='#FFFFFF')
    plt.axhline(y=0.5, color='k', linestyle='--')
    plt.title('MIA AUC Against Quantity of Training Data',
              fontname=font,
              fontsize=title_size)
    plt.xlabel('Quantity of Training Data', fontname=font, fontsize=axes_size)
    plt.ylabel('MIA AUC', fontname=font, fontsize=axes_size)
    plt.xlim([0, 40000])
    plt.xticks(fontname=font, fontsize=axes_size)
    plt.yticks(fontname=font, fontsize=axes_size)
    plt.legend(labels=['AUC', 'Percentage Error', 'Baseline Accuracy'],
               fontsize=legend_size)

    plt.subplot(2, 2, 3)
    avg_ref_set = np.mean(reference_set, axis=0)
    avg_synth_set = np.mean(synth_set, axis=0)

    # Plotting the first x_set ECG KDE against the averages to show which it may be closer to, along with printing
    # its classification. Also plotting the last x_test (not from training) and assignment to see if it is closer to ref
    sns.kdeplot(avg_synth_set, fill=True, color=royal_blue)
    sns.kdeplot(x_test[0], color=dark_royal_blue)  # CHANGE
    sns.kdeplot(avg_ref_set, fill=True, color=red)
    sns.kdeplot(x_test[len(x_test) - 20], color=dark_red)
    plt.xlim([0, 1.2])

    plt.xticks(fontname=font, fontsize=axes_size)
    plt.yticks(fontname=font, fontsize=axes_size)
    plt.title('Densities of ECG Amplitudes for 500 Training Samples',
              # for Prediction Success Comparison - changed from suptitle
              fontname=font,
              fontsize=title_size)

    plt.xlabel('ECG Amplitude', fontname=font, fontsize=axes_size)
    plt.ylabel('Density', fontname=font, fontsize=axes_size)
    plt.legend(labels=['Synthetic Data',
                       f'Training Sample | Prediction = {y_pred[0]}',
                       'Reference Data',
                       f'Reference Sample | Prediction = {y_pred[len(y_pred) - 20]}'],
               fontsize=legend_size)

    mem_set = pd.read_csv(f'Saved Data/{FOLDER_NAME}/{SPECIFIC_NAME_2}/mem_set.csv').drop(['Unnamed: 0'],
                                                                                          axis=1).to_numpy(
        dtype=np.float32)
    reference_set = pd.read_csv(f'Saved Data/{FOLDER_NAME}/{SPECIFIC_NAME_2}/ref_set.csv').drop(['Unnamed: 0'],
                                                                                                axis=1).to_numpy(
        dtype=np.float32)
    synth_set = pd.read_csv(f'Saved Data/{FOLDER_NAME}/{SPECIFIC_NAME_2}/synth_set.csv').drop(['Unnamed: 0'],
                                                                                              axis=1).to_numpy(
        dtype=np.float32)
    x_test = pd.read_csv(f'Saved Data/{FOLDER_NAME}/{SPECIFIC_NAME_2}/x_test.csv').drop(['Unnamed: 0'],
                                                                                        axis=1).to_numpy(
        dtype=np.float32)
    y_test = pd.read_csv(f'Saved Data/{FOLDER_NAME}/{SPECIFIC_NAME_2}/y_test.csv').drop(['Unnamed: 0'],
                                                                                        axis=1).to_numpy(
        dtype=np.bool_).flatten()
    y_pred = pd.read_csv(f'Saved Data/{FOLDER_NAME}/{SPECIFIC_NAME_2}/y_pred.csv').drop(['Unnamed: 0'],
                                                                                        axis=1).to_numpy(
        dtype=np.bool_).flatten()

    plt.subplot(2, 2, 4)  # divide as 2x2, plot top right
    # plt.plt(...)
    avg_ref_set = np.mean(reference_set, axis=0)
    avg_synth_set = np.mean(synth_set, axis=0)

    # Plotting the first x_set ECG KDE against the averages to show which it may be closer to, along with printing
    # its classification. Also plotting the last x_test (not from training) and assignment to see if it is closer to ref
    sns.kdeplot(avg_synth_set, fill=True, color=royal_blue)
    sns.kdeplot(x_test[0], color=dark_royal_blue)  # CHANGE
    sns.kdeplot(avg_ref_set, fill=True, color=red)
    sns.kdeplot(x_test[len(x_test) - 1], color=dark_red)
    plt.xlim([0, 1.2])

    plt.xticks(fontname=font, fontsize=axes_size)
    plt.yticks(fontname=font, fontsize=axes_size)
    plt.title('Densities of ECG Amplitudes for 20,000 Training Samples',
              # for Prediction Success Comparison - changed from suptitle
              fontname=font,
              fontsize=title_size)

    plt.xlabel('ECG Amplitude', fontname=font, fontsize=axes_size)
    plt.ylabel('Density', fontname=font, fontsize=axes_size)
    plt.legend(labels=['Synthetic Data',
                       f'Training Sample | Prediction = {y_pred[0]}',
                       'Reference Data',
                       f'Reference Sample | Prediction = {y_pred[len(y_pred) - 1]}'],
               fontsize=legend_size)
    plt.tight_layout()
    plt.show()


def synthetic():
    FOLDER_NAME = "train 1000, ref 1000, synth 5000, epochs 25/Synth Data Tests"
    RESULTS = "synth_results.csv"

    # For plotting specific results
    SPECIFIC_NAME_1 = "TVAE_31-05-24-1628_1000_1000_7500_25_kde_mia_domias_evaluation_scott"  # Of specific graph
    SPECIFIC_NAME_2 = "TVAE_31-05-24-1628_1000_1000_40000_25_kde_mia_domias_evaluation_scott"  # Of specific graph

    all_results = pd.read_csv(f'Saved Data/{FOLDER_NAME}/{RESULTS}').drop(
        ['Generator', 'Date', 'MIA Method', 'Bandwidth'], axis=1).to_numpy(dtype=np.float32)

    mem_set = pd.read_csv(f'Saved Data/{FOLDER_NAME}/{SPECIFIC_NAME_1}/mem_set.csv').drop(['Unnamed: 0'],
                                                                                          axis=1).to_numpy(
        dtype=np.float32)
    reference_set = pd.read_csv(f'Saved Data/{FOLDER_NAME}/{SPECIFIC_NAME_1}/ref_set.csv').drop(['Unnamed: 0'],
                                                                                                axis=1).to_numpy(
        dtype=np.float32)
    synth_set = pd.read_csv(f'Saved Data/{FOLDER_NAME}/{SPECIFIC_NAME_1}/synth_set.csv').drop(['Unnamed: 0'],
                                                                                              axis=1).to_numpy(
        dtype=np.float32)
    x_test = pd.read_csv(f'Saved Data/{FOLDER_NAME}/{SPECIFIC_NAME_1}/x_test.csv').drop(['Unnamed: 0'],
                                                                                        axis=1).to_numpy(
        dtype=np.float32)
    y_test = pd.read_csv(f'Saved Data/{FOLDER_NAME}/{SPECIFIC_NAME_1}/y_test.csv').drop(['Unnamed: 0'],
                                                                                        axis=1).to_numpy(
        dtype=np.bool_).flatten()
    y_pred = pd.read_csv(f'Saved Data/{FOLDER_NAME}/{SPECIFIC_NAME_1}/y_pred.csv').drop(['Unnamed: 0'],
                                                                                        axis=1).to_numpy(
        dtype=np.bool_).flatten()

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot Top
    plt.subplot(2, 1, 1)  # divide as 2x1, plot top
    plt.plot(all_results[:, 2], all_results[:, 5], color=dark_royal_blue)  # AUC
    all_results2 = pd.read_csv(
        'Saved Data/train 1000, ref 1000, synth 5000, epochs 25/Synth Data Tests 2/synth_results2.csv').drop(
        ['Generator', 'Date', 'MIA Method', 'Bandwidth'], axis=1).to_numpy(dtype=np.float32)
    error = abs(all_results2[:, 5] - all_results[:, 5]) * all_results[:, 5]
    y_error_pos = all_results[:, 5] + error
    y_error_neg = all_results[:, 5] - error
    plt.fill_between(all_results[:, 2], all_results[:, 5] - error, all_results[:, 5] + error, color=royal_blue,
                     alpha=0.5, edgecolor='#FFFFFF')
    plt.axhline(y=0.5, color='k', linestyle='--')
    plt.title('MIA AUC Against Quantity of Synthetic Data Produced',
              fontname=font,
              fontsize=title_size)
    plt.xlabel('Quantity of Synthetic Data', fontname=font, fontsize=axes_size)
    plt.ylabel('MIA AUC', fontname=font, fontsize=axes_size)
    plt.xlim([0, 40000])
    plt.ylim([0.49, 0.55])
    plt.xticks(fontname=font, fontsize=axes_size)
    plt.yticks(fontname=font, fontsize=axes_size)
    plt.legend(labels=['AUC', 'Percentage Error', 'Baseline Accuracy'],
               fontsize=legend_size)

    plt.subplot(2, 2, 3)
    avg_ref_set = np.mean(reference_set, axis=0)
    avg_synth_set = np.mean(synth_set, axis=0)

    # Plotting the first x_set ECG KDE against the averages to show which it may be closer to, along with printing
    # its classification. Also plotting the last x_test (not from training) and assignment to see if it is closer to ref
    sns.kdeplot(avg_synth_set, fill=True, color=royal_blue)
    sns.kdeplot(x_test[0], color=dark_royal_blue)  # CHANGE
    sns.kdeplot(avg_ref_set, fill=True, color=red)
    sns.kdeplot(x_test[len(x_test) - 1], color=dark_red)
    plt.xlim([0, 1.2])

    plt.xticks(fontname=font, fontsize=axes_size)
    plt.yticks(fontname=font, fontsize=axes_size)
    plt.title('Densities of ECG Amplitudes for 7,500 Synthetic Samples',
              # for Prediction Success Comparison - changed from suptitle
              fontname=font,
              fontsize=title_size)

    plt.xlabel('ECG Amplitude', fontname=font, fontsize=axes_size)
    plt.ylabel('Density', fontname=font, fontsize=axes_size)
    plt.legend(labels=['Synthetic Data',
                       f'Training Sample | Prediction = {y_pred[0]}',
                       'Reference Data',
                       f'Reference Sample | Prediction = {y_pred[len(y_pred) - 1]}'],
               fontsize=legend_size)

    mem_set = pd.read_csv(f'Saved Data/{FOLDER_NAME}/{SPECIFIC_NAME_2}/mem_set.csv').drop(['Unnamed: 0'],
                                                                                          axis=1).to_numpy(
        dtype=np.float32)
    reference_set = pd.read_csv(f'Saved Data/{FOLDER_NAME}/{SPECIFIC_NAME_2}/ref_set.csv').drop(['Unnamed: 0'],
                                                                                                axis=1).to_numpy(
        dtype=np.float32)
    synth_set = pd.read_csv(f'Saved Data/{FOLDER_NAME}/{SPECIFIC_NAME_2}/synth_set.csv').drop(['Unnamed: 0'],
                                                                                              axis=1).to_numpy(
        dtype=np.float32)
    x_test = pd.read_csv(f'Saved Data/{FOLDER_NAME}/{SPECIFIC_NAME_2}/x_test.csv').drop(['Unnamed: 0'],
                                                                                        axis=1).to_numpy(
        dtype=np.float32)
    y_test = pd.read_csv(f'Saved Data/{FOLDER_NAME}/{SPECIFIC_NAME_2}/y_test.csv').drop(['Unnamed: 0'],
                                                                                        axis=1).to_numpy(
        dtype=np.bool_).flatten()
    y_pred = pd.read_csv(f'Saved Data/{FOLDER_NAME}/{SPECIFIC_NAME_2}/y_pred.csv').drop(['Unnamed: 0'],
                                                                                        axis=1).to_numpy(
        dtype=np.bool_).flatten()

    plt.subplot(2, 2, 4)  # divide as 2x2, plot top right
    # plt.plt(...)
    avg_ref_set = np.mean(reference_set, axis=0)
    avg_synth_set = np.mean(synth_set, axis=0)

    # Plotting the first x_set ECG KDE against the averages to show which it may be closer to, along with printing
    # its classification. Also plotting the last x_test (not from training) and assignment to see if it is closer to ref
    sns.kdeplot(avg_synth_set, fill=True, color=royal_blue)
    sns.kdeplot(x_test[26], color=dark_royal_blue)  # CHANGE
    sns.kdeplot(avg_ref_set, fill=True, color=red)
    sns.kdeplot(x_test[len(x_test) - 1], color=dark_red)
    plt.xlim([0, 1.2])

    plt.xticks(fontname=font, fontsize=axes_size)
    plt.yticks(fontname=font, fontsize=axes_size)
    plt.title('Densities of ECG Amplitudes for 40,000 Synthetic Samples',
              # for Prediction Success Comparison - changed from suptitle
              fontname=font,
              fontsize=title_size)

    plt.xlabel('ECG Amplitude', fontname=font, fontsize=axes_size)
    plt.ylabel('Density', fontname=font, fontsize=axes_size)
    plt.legend(labels=['Synthetic Data',
                       f'Training Sample | Prediction = {y_pred[26]}',
                       'Reference Data',
                       f'Reference Sample | Prediction = {y_pred[len(y_pred) - 1]}'],
               fontsize=legend_size)
    plt.tight_layout()
    plt.show()


def ecg_epochs():
    FOLDER_NAME = "Epoch Tests"
    RESULTS = "epoch_results.csv"

    # For plotting specific results
    SPECIFIC_NAME_1 = "TVAE_07-05-24-1244_1000_1000_5000_25_kde_mia_domias_evaluation_scott"  # Of specific graph
    SPECIFIC_NAME_2 = "TVAE_07-05-24-1432_1000_1000_5000_1500_kde_mia_domias_evaluation_scott"  # Of specific graph

    all_results = pd.read_csv(f'Saved Data/{FOLDER_NAME}/{RESULTS}').drop(
        ['Generator', 'Date', 'MIA Method', 'Bandwidth'], axis=1).to_numpy(dtype=np.float32)

    mem_set = pd.read_csv(f'Saved Data/{FOLDER_NAME}/{SPECIFIC_NAME_1}/mem_set.csv').drop(['Unnamed: 0'],
                                                                                          axis=1).to_numpy(
        dtype=np.float32)
    reference_set = pd.read_csv(f'Saved Data/{FOLDER_NAME}/{SPECIFIC_NAME_1}/ref_set.csv').drop(['Unnamed: 0'],
                                                                                                axis=1).to_numpy(
        dtype=np.float32)
    synth_set = pd.read_csv(f'Saved Data/{FOLDER_NAME}/{SPECIFIC_NAME_1}/synth_set.csv').drop(['Unnamed: 0'],
                                                                                              axis=1).to_numpy(
        dtype=np.float32)
    x_test = pd.read_csv(f'Saved Data/{FOLDER_NAME}/{SPECIFIC_NAME_1}/x_test.csv').drop(['Unnamed: 0'],
                                                                                        axis=1).to_numpy(
        dtype=np.float32)
    y_test = pd.read_csv(f'Saved Data/{FOLDER_NAME}/{SPECIFIC_NAME_1}/y_test.csv').drop(['Unnamed: 0'],
                                                                                        axis=1).to_numpy(
        dtype=np.bool_).flatten()
    y_pred = pd.read_csv(f'Saved Data/{FOLDER_NAME}/{SPECIFIC_NAME_1}/y_pred.csv').drop(['Unnamed: 0'],
                                                                                        axis=1).to_numpy(
        dtype=np.bool_).flatten()

    fig, axes = plt.subplots(1, 2, figsize=(22, 6))

    # Plotting ECG Signals - first in each array
    plt.subplot(1, 2, 1)
    plt.plot(x_test[0], color=dark_turquoise)  # CHANGE
    plt.plot(synth_set[0], color=red)
    plt.plot(reference_set[0], color=blue)

    plt.title('Synthetic ECG Using 25 Epochs',
              fontname=font,
              fontsize=title_size)

    # plt.title(f'Synthesiser: {info[0]} | Training Set Size: {info[1]} | '
    #           f'Reference Set Size: {info[2]} | '
    #           f'Synthetic Set Size: {info[3]} | Epochs: {info[4]} | '
    #           f'Bandwidth: {info[5]}',
    #           fontname=font,
    #           fontsize=subtitle_size)

    plt.xlabel('Sample', fontname=font, fontsize=axes_size)
    plt.ylabel('ECG Amplitude', fontname=font, fontsize=axes_size)

    plt.xticks(fontname=font, fontsize=axes_size)
    plt.yticks(fontname=font, fontsize=axes_size)

    plt.legend(labels=['Training ECG', 'Synthetic ECG', 'Reference ECG'],
               fontsize=legend_size)

    mem_set = pd.read_csv(f'Saved Data/{FOLDER_NAME}/{SPECIFIC_NAME_2}/mem_set.csv').drop(['Unnamed: 0'],
                                                                                          axis=1).to_numpy(
        dtype=np.float32)
    reference_set = pd.read_csv(f'Saved Data/{FOLDER_NAME}/{SPECIFIC_NAME_2}/ref_set.csv').drop(['Unnamed: 0'],
                                                                                                axis=1).to_numpy(
        dtype=np.float32)
    synth_set = pd.read_csv(f'Saved Data/{FOLDER_NAME}/{SPECIFIC_NAME_2}/synth_set.csv').drop(['Unnamed: 0'],
                                                                                              axis=1).to_numpy(
        dtype=np.float32)
    x_test = pd.read_csv(f'Saved Data/{FOLDER_NAME}/{SPECIFIC_NAME_2}/x_test.csv').drop(['Unnamed: 0'],
                                                                                        axis=1).to_numpy(
        dtype=np.float32)
    y_test = pd.read_csv(f'Saved Data/{FOLDER_NAME}/{SPECIFIC_NAME_2}/y_test.csv').drop(['Unnamed: 0'],
                                                                                        axis=1).to_numpy(
        dtype=np.bool_).flatten()
    y_pred = pd.read_csv(f'Saved Data/{FOLDER_NAME}/{SPECIFIC_NAME_2}/y_pred.csv').drop(['Unnamed: 0'],
                                                                                        axis=1).to_numpy(
        dtype=np.bool_).flatten()

    plt.subplot(1, 2, 2)
    plt.plot(x_test[0], color=dark_turquoise)  # CHANGE
    plt.plot(synth_set[0], color=red)
    plt.plot(reference_set[0], color=blue)

    plt.title('Synthetic ECG Using 1,500 Epochs',
              fontname=font,
              fontsize=title_size)

    # plt.title(f'Synthesiser: {info[0]} | Training Set Size: {info[1]} | '
    #           f'Reference Set Size: {info[2]} | '
    #           f'Synthetic Set Size: {info[3]} | Epochs: {info[4]} | '
    #           f'Bandwidth: {info[5]}',
    #           fontname=font,
    #           fontsize=subtitle_size)

    plt.xlabel('Sample', fontname=font, fontsize=axes_size)
    plt.ylabel('ECG Amplitude', fontname=font, fontsize=axes_size)

    plt.xticks(fontname=font, fontsize=axes_size)
    plt.yticks(fontname=font, fontsize=axes_size)

    plt.legend(labels=['Training ECG', 'Synthetic ECG', 'Reference ECG'],
               fontsize=legend_size)

    plt.show()


def ecg_training():
    FOLDER_NAME = "train 1000, ref 1000, synth 5000, epochs 25/Training Set Tests"
    RESULTS = "training_results1.csv"

    # For plotting specific results
    SPECIFIC_NAME_1 = "TVAE_16-05-24-0921_500_1000_5000_25_kde_mia_domias_evaluation_scott"  # Of specific graph
    SPECIFIC_NAME_2 = "TVAE_16-05-24-0921_20000_1000_5000_25_kde_mia_domias_evaluation_scott"  # Of specific graph

    all_results = pd.read_csv(f'Saved Data/{FOLDER_NAME}/{RESULTS}').drop(
        ['Generator', 'Date', 'MIA Method', 'Bandwidth'], axis=1).to_numpy(dtype=np.float32)

    mem_set = pd.read_csv(f'Saved Data/{FOLDER_NAME}/{SPECIFIC_NAME_1}/mem_set.csv').drop(['Unnamed: 0'],
                                                                                          axis=1).to_numpy(
        dtype=np.float32)
    reference_set = pd.read_csv(f'Saved Data/{FOLDER_NAME}/{SPECIFIC_NAME_1}/ref_set.csv').drop(['Unnamed: 0'],
                                                                                                axis=1).to_numpy(
        dtype=np.float32)
    synth_set = pd.read_csv(f'Saved Data/{FOLDER_NAME}/{SPECIFIC_NAME_1}/synth_set.csv').drop(['Unnamed: 0'],
                                                                                              axis=1).to_numpy(
        dtype=np.float32)
    x_test = pd.read_csv(f'Saved Data/{FOLDER_NAME}/{SPECIFIC_NAME_1}/x_test.csv').drop(['Unnamed: 0'],
                                                                                        axis=1).to_numpy(
        dtype=np.float32)
    y_test = pd.read_csv(f'Saved Data/{FOLDER_NAME}/{SPECIFIC_NAME_1}/y_test.csv').drop(['Unnamed: 0'],
                                                                                        axis=1).to_numpy(
        dtype=np.bool_).flatten()
    y_pred = pd.read_csv(f'Saved Data/{FOLDER_NAME}/{SPECIFIC_NAME_1}/y_pred.csv').drop(['Unnamed: 0'],
                                                                                        axis=1).to_numpy(
        dtype=np.bool_).flatten()

    fig, axes = plt.subplots(1, 2, figsize=(22, 6))

    # Plotting ECG Signals - first in each array
    plt.subplot(1, 2, 1)
    plt.plot(x_test[0], color=dark_turquoise)  # CHANGE
    plt.plot(synth_set[0], color=red)
    plt.plot(reference_set[0], color=blue)

    plt.title('Synthetic ECG Using 500 Training Samples',
              fontname=font,
              fontsize=title_size)

    # plt.title(f'Synthesiser: {info[0]} | Training Set Size: {info[1]} | '
    #           f'Reference Set Size: {info[2]} | '
    #           f'Synthetic Set Size: {info[3]} | Epochs: {info[4]} | '
    #           f'Bandwidth: {info[5]}',
    #           fontname=font,
    #           fontsize=subtitle_size)

    plt.xlabel('Sample', fontname=font, fontsize=axes_size)
    plt.ylabel('ECG Amplitude', fontname=font, fontsize=axes_size)

    plt.xticks(fontname=font, fontsize=axes_size)
    plt.yticks(fontname=font, fontsize=axes_size)

    plt.legend(labels=['Training ECG', 'Synthetic ECG', 'Reference ECG'],
               fontsize=legend_size)

    mem_set = pd.read_csv(f'Saved Data/{FOLDER_NAME}/{SPECIFIC_NAME_2}/mem_set.csv').drop(['Unnamed: 0'],
                                                                                          axis=1).to_numpy(
        dtype=np.float32)
    reference_set = pd.read_csv(f'Saved Data/{FOLDER_NAME}/{SPECIFIC_NAME_2}/ref_set.csv').drop(['Unnamed: 0'],
                                                                                                axis=1).to_numpy(
        dtype=np.float32)
    synth_set = pd.read_csv(f'Saved Data/{FOLDER_NAME}/{SPECIFIC_NAME_2}/synth_set.csv').drop(['Unnamed: 0'],
                                                                                              axis=1).to_numpy(
        dtype=np.float32)
    x_test = pd.read_csv(f'Saved Data/{FOLDER_NAME}/{SPECIFIC_NAME_2}/x_test.csv').drop(['Unnamed: 0'],
                                                                                        axis=1).to_numpy(
        dtype=np.float32)
    y_test = pd.read_csv(f'Saved Data/{FOLDER_NAME}/{SPECIFIC_NAME_2}/y_test.csv').drop(['Unnamed: 0'],
                                                                                        axis=1).to_numpy(
        dtype=np.bool_).flatten()
    y_pred = pd.read_csv(f'Saved Data/{FOLDER_NAME}/{SPECIFIC_NAME_2}/y_pred.csv').drop(['Unnamed: 0'],
                                                                                        axis=1).to_numpy(
        dtype=np.bool_).flatten()

    plt.subplot(1, 2, 2)
    plt.plot(x_test[0], color=dark_turquoise)  # CHANGE
    plt.plot(synth_set[0], color=red)
    plt.plot(reference_set[0], color=blue)

    plt.title('Synthetic ECG Using 20,000 Training Samples',
              fontname=font,
              fontsize=title_size)

    # plt.title(f'Synthesiser: {info[0]} | Training Set Size: {info[1]} | '
    #           f'Reference Set Size: {info[2]} | '
    #           f'Synthetic Set Size: {info[3]} | Epochs: {info[4]} | '
    #           f'Bandwidth: {info[5]}',
    #           fontname=font,
    #           fontsize=subtitle_size)

    plt.xlabel('Sample', fontname=font, fontsize=axes_size)
    plt.ylabel('ECG Amplitude', fontname=font, fontsize=axes_size)

    plt.xticks(fontname=font, fontsize=axes_size)
    plt.yticks(fontname=font, fontsize=axes_size)

    plt.legend(labels=['Training ECG', 'Synthetic ECG', 'Reference ECG'],
               fontsize=legend_size)

    plt.show()


def ecg_par():
    FOLDER_NAME = "PAR"
    RESULTS = "par_results.csv"

    # For plotting specific results
    SPECIFIC_NAME_1 = "PAR_02-06-24-1258_1000_1000_1_25_kde_mia_domias_evaluation_scott"  # Of specific graph

    mem_set = pd.read_csv(f'Saved Data/{FOLDER_NAME}/{SPECIFIC_NAME_1}/mem_set.csv').drop(['Unnamed: 0'],
                                                                                          axis=1).to_numpy(
        dtype=np.float32)
    reference_set = pd.read_csv(f'Saved Data/{FOLDER_NAME}/{SPECIFIC_NAME_1}/ref_set.csv').drop(['Unnamed: 0'],
                                                                                                axis=1).to_numpy(
        dtype=np.float32)
    synth_set = pd.read_csv(f'Saved Data/{FOLDER_NAME}/{SPECIFIC_NAME_1}/synth_set.csv').drop(['Unnamed: 0'],
                                                                                              axis=1).to_numpy(
        dtype=np.float32)
    x_test = pd.read_csv(f'Saved Data/{FOLDER_NAME}/{SPECIFIC_NAME_1}/x_test.csv').drop(['Unnamed: 0'],
                                                                                        axis=1).to_numpy(
        dtype=np.float32)
    y_test = pd.read_csv(f'Saved Data/{FOLDER_NAME}/{SPECIFIC_NAME_1}/y_test.csv').drop(['Unnamed: 0'],
                                                                                        axis=1).to_numpy(
        dtype=np.bool_).flatten()
    y_pred = pd.read_csv(f'Saved Data/{FOLDER_NAME}/{SPECIFIC_NAME_1}/y_pred.csv').drop(['Unnamed: 0'],
                                                                                        axis=1).to_numpy(
        dtype=np.bool_).flatten()

    # Plotting ECG Signals - first in each array
    plt.plot(x_test[0], color=dark_turquoise)  # CHANGE
    plt.plot(synth_set[0], color=red)
    plt.plot(reference_set[0], color=blue)

    plt.title('ECG Recording Using PAR Synthesiser',
              fontname=font,
              fontsize=title_size)

    # plt.title(f'Synthesiser: {info[0]} | Training Set Size: {info[1]} | '
    #           f'Reference Set Size: {info[2]} | '
    #           f'Synthetic Set Size: {info[3]} | Epochs: {info[4]} | '
    #           f'Bandwidth: {info[5]}',
    #           fontname=font,
    #           fontsize=subtitle_size)

    plt.xlabel('Sample', fontname=font, fontsize=axes_size)
    plt.ylabel('ECG Amplitude', fontname=font, fontsize=axes_size)

    plt.xticks(fontname=font, fontsize=axes_size)
    plt.yticks(fontname=font, fontsize=axes_size)

    plt.legend(labels=['Training ECG', 'Synthetic ECG', 'Reference ECG'],
               fontsize=legend_size)

    plt.show()


if __name__ == '__main__':
    ecg_training()
