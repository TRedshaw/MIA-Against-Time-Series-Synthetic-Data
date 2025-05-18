# Third Party
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from matplotlib.ticker import PercentFormatter
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

title_size = 32
subtitle_size = 20
axes_size = 32


def plot_ecg_1(synthetic_data, reference_data, test_data, info):
    # Plotting ECG Signals - first in each array
    plt.plot(test_data[0], color=dark_turquoise)  # CHANGE
    plt.plot(synthetic_data[0], color=red)
    plt.plot(reference_data[0], color=blue)

    plt.suptitle('ECG Recordings',
                 fontname=font,
                 fontsize=title_size)

    plt.title(f'Synthesiser: {info[0]} | Training Set Size: {info[1]} | '
              f'Reference Set Size: {info[2]} | '
              f'Synthetic Set Size: {info[3]} | Epochs: {info[4]} | '
              f'Bandwidth: {info[5]}',
              fontname=font,
              fontsize=subtitle_size)

    plt.xlabel('Sample', fontname=font, fontsize=axes_size)
    plt.ylabel('ECG Value', fontname=font, fontsize=axes_size)

    plt.xticks(fontname=font, fontsize=axes_size)
    plt.yticks(fontname=font, fontsize=axes_size)

    plt.legend(labels=['Training ECG', 'Synthetic ECG', 'Population ECG'],
               fontsize=20)

    plt.show()


# def plot_all_kde(synthetic_data, reference_data):
#     # Plotting KDE of synth and ref set (synth = filled) - Plots every KDE
#     sns.kdeplot(synthetic_data.transpose(1, 0), fill=True)
#     sns.kdeplot(reference_data.transpose(1, 0))
#     plt.show()
#     pass


# def plot_average_kde(synthetic_data, reference_data):
#     avg_ref_set = np.mean(reference_data, axis=0)
#     avg_synth_set = np.mean(synthetic_data, axis=0)
#
#     # Plotting KDE of synth and ref set (ref = filled) = IS THE AVERAGE - IS THIS CORRECT
#     sns.kdeplot(avg_ref_set, fill=True)
#     sns.kdeplot(avg_synth_set)
#     plt.legend(labels=['ref set', 'synth set'])
#     plt.show()


def plot_average_kde_vs_training_and_non_kde(synthetic_data, reference_data, test_data, membership_prediction, info):
    """Used to compare the average KDE (which I think is best KDE representation), against the KDE of an ECG signal
    used in training, and one that wasn't. It is then possible to compare the KDE of a training or non-training
    member to the synth and reference KDEs, and compare which is closer and then the assignment of if the program
    thinks it is part of the training data. It will likely show True if it is close to synth, showing the program
    acts as it should. I COULD THEN COMPARE THE PREDICTION to the actual for that one to support how time sensitivity
    is needed (for one that gets it wrong and plot the ECG's with it to show even if KDE is similar, ECG might not be,
    showing need for time sensitivity for accurate prediction, and even more for specific MIA."""

    avg_ref_set = np.mean(reference_data, axis=0)
    avg_synth_set = np.mean(synthetic_data, axis=0)

    # Plotting the first x_set ECG KDE against the averages to show which it may be closer to, along with printing
    # its classification. Also plotting the last x_test (not from training) and assignment to see if it is closer to ref
    ax = sns.kdeplot(avg_synth_set, fill=True, color=royal_blue)
    # ax.yaxis.set_major_formatter(PercentFormatter(1 / 5))
    # sns.kdeplot(test_data[0], color=dark_royal_blue)  # CHANGE
    # sns.kdeplot(avg_ref_set, fill=True, color=red)
    # sns.kdeplot(test_data[len(test_data) - 20], color=dark_red)

    plt.title('Probability Densities of ECG Values',  # for Prediction Success Comparison - changed from suptitle
                 fontname=font,
                 fontsize=title_size)

    # plt.title(f'Synthesiser: {info[0]} | Training Set Size: {info[1]} | '
    #           f'Reference Set Size: {info[2]} | '
    #           f'Synthetic Set Size: {info[3]} | Epochs: {info[4]} | '
    #           f'Bandwidth: {info[5]}',
    #           fontname=font,
    #           fontsize=subtitle_size)

    plt.xlabel('ECG Value', fontname=font, fontsize=axes_size)
    plt.ylabel('Density', fontname=font, fontsize=axes_size)

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.xlim([0, 1.2])

    plt.xticks(fontname=font, fontsize=axes_size)
    plt.yticks(fontname=font, fontsize=axes_size)

    plt.legend(labels=['Synthetic Data',
                       f'Training Sample PDF | Assignment = {membership_prediction[0]}',
                       'Reference Data',
                       f'Reference Sample PDF | Assignment = {membership_prediction[len(membership_prediction) - 20]}'],
               fontsize=20)
    plt.show()


def plot_kde_difference(synthetic_data, reference_data, test_data, membership_prediction, info):

    avg_ref_set = np.mean(reference_data, axis=0)
    avg_synth_set = np.mean(synthetic_data, axis=0)
    # avg_difference = gaussian_filter1d(abs(avg_synth_set-avg_ref_set)*10, sigma=3)

    X_plot = np.linspace(0, 488)
    density_gen = KernelDensity(kernel='gaussian', bandwidth='scott').fit(synth_set)
    density_data = KernelDensity(kernel='gaussian', bandwidth='scott').fit(reference_set)
    plt.plot(density_data)
    plt.show()

    # Plotting the first x_set ECG KDE against the averages to show which it may be closer to, along with printing
    # its classification. Also plotting the last x_test (not from training) and assignment to see if it is closer to ref
    sns.kdeplot(avg_synth_set, fill=True, color=royal_blue)
    sns.kdeplot(test_data[500], color=dark_royal_blue)
    sns.kdeplot(avg_ref_set, fill=True, color=purple)
    sns.kdeplot(test_data[len(test_data) - 1], color=dark_purple)
    # plt.plot(avg_difference, color=turquoise)


    plt.suptitle('Probability Densities of ECG Values for Prediction Success Comparison',
                 fontname=font,
                 fontsize=title_size)

    plt.title(f'Synthesiser: {info[0]} | Training Set Size: {info[1]} | '
              f'Reference Set Size: {info[2]} | '
              f'Synthetic Set Size: {info[3]} | Epochs: {info[4]} | '
              f'Bandwidth: {info[5]}',
              fontname=font,
              fontsize=subtitle_size)

    plt.xlabel('ECG Value', fontname=font, fontsize=axes_size)
    plt.ylabel('Density', fontname=font, fontsize=axes_size)

    plt.ylim([0, 1.2])

    plt.xticks(fontname=font, fontsize=axes_size)
    plt.yticks(fontname=font, fontsize=axes_size)

    plt.legend(labels=[
                        'Synthetic Data PDF',
                        f'Training Sample PDF | Assignment = {membership_prediction[0]}',
                        'Population Data PDF',
                        f'Population Sample PDF | Assignment = {membership_prediction[len(membership_prediction) - 1]}',
                        'Difference in Synth and Population Data PDF'
                       ],
               fontsize=20)
    plt.show()


# def plot_first_kde_vs_training_and_non_kde(synthetic_data, reference_data, membership_prediction):
#     # Could also plot the first KDE's - and comapre to first and last x test instead of avg
#     sns.kdeplot(reference_data[0, :], fill=True)
#     sns.kdeplot(synthetic_data[0, :])
#     plt.legend(labels=['first ref set kde',
#                        'first synth set kde',
#                        f'x_test[0] (from training) | Assignment = {membership_prediction[0]}',
#                        f'x_test[end] (unseen) | Assignment = {membership_prediction[len(membership_prediction) - 1]}'])
#     plt.show()


def plot_epochs(all_results):
    plt.plot(all_results[:, 3], all_results[:, 5], color=royal_blue)  # AUC test 1

    all_results = pd.read_csv(f'Saved Data/Epoch Tests 2/epoch_results2.csv').drop(
        ['Generator', 'Date', 'MIA Method', 'Bandwidth'], axis=1).to_numpy(dtype=np.float32)
    plt.plot(all_results[:, 3], all_results[:, 5], color=dark_royal_blue)  # AUC test 1


    plt.title('AUC against Number of Epochs',
                 fontname=font,
                 fontsize=title_size)

    plt.xlabel('Number of Epochs', fontname=font, fontsize=axes_size)
    plt.ylabel('AUC', fontname=font, fontsize=axes_size)

    plt.xscale('log')

    plt.xticks(fontname=font, fontsize=axes_size)
    plt.yticks(fontname=font, fontsize=axes_size)

    plt.legend(labels=['AUC Test 1', 'AUC Test 2'],
               fontsize=20)

    plt.show()


def plot_epochs_error(all_results):
    plt.plot(all_results[:, 3], all_results[:, 5], color=dark_royal_blue)  # AUC test 1

    all_results2 = pd.read_csv(f'Saved Data/Epoch Tests 2/epoch_results2.csv').drop(
        ['Generator', 'Date', 'MIA Method', 'Bandwidth'], axis=1).to_numpy(dtype=np.float32)

    error = abs(all_results2[:, 5]-all_results[:, 5])*all_results[:, 5]
    y_error_pos = all_results[:, 5] + error
    y_error_neg = all_results[:, 5] - error

    plt.fill_between(all_results[:, 3], all_results[:, 5] - error, all_results[:, 5] + error, color=royal_blue, alpha=0.5, edgecolor='#FFFFFF')

    plt.axhline(y=0.5, color='k', linestyle='--')

    plt.xlim([5, 10000])
    plt.ylim([0.48, 0.72])

    plt.title('AUC Against Number of Epochs',
                 fontname=font,
                 fontsize=title_size)

    plt.xlabel('Number of Epochs', fontname=font, fontsize=axes_size)
    plt.ylabel('AUC', fontname=font, fontsize=axes_size)

    plt.xscale('log')

    plt.xticks(fontname=font, fontsize=axes_size)
    plt.yticks(fontname=font, fontsize=axes_size)

    plt.legend(labels=['AUC', 'Percentage Error', 'Baseline Accuracy'],
               fontsize=20,
               loc='upper left')

    plt.show()


def plot_train_error(all_results):
    plt.plot(all_results[:, 0], all_results[:, 5], color=dark_royal_blue)  # AUC

    all_results2 = pd.read_csv(f'Saved Data/train 1000, ref 1000, synth 5000, epochs 25/Training Set Tests 2/training_results2.csv').drop(
        ['Generator', 'Date', 'MIA Method', 'Bandwidth'], axis=1).to_numpy(dtype=np.float32)

    error = abs(all_results2[:, 5] - all_results[:, 5]) * all_results[:, 5]
    y_error_pos = all_results[:, 5] + error
    y_error_neg = all_results[:, 5] - error
    plt.fill_between(all_results[:, 0], all_results[:, 5] - error, all_results[:, 5] + error, color=royal_blue,
                     alpha=0.5, edgecolor='#FFFFFF')

    plt.axhline(y=0.5, color='k', linestyle='--')

    plt.title('AUC Against Amount of Training Data',
                 fontname=font,
                 fontsize=title_size)

    plt.xlabel('Amount of Training Data', fontname=font, fontsize=axes_size)
    plt.ylabel('AUC', fontname=font, fontsize=axes_size)

    plt.xlim([0,40000])

    plt.xticks(fontname=font, fontsize=axes_size)
    plt.yticks(fontname=font, fontsize=axes_size)

    plt.legend(labels=['AUC', 'Percentage Error', 'Baseline Accuracy'],
               fontsize=20)

    plt.show()


def plot_ref(all_results):
    plt.plot(all_results[:, 1], all_results[:, 5], color=royal_blue)  # AUC

    all_results2 = pd.read_csv(
        f'Saved Data/train 1000, ref 1000, synth 5000, epochs 25/Reference Tests 2/reference_results2.csv').drop(
        ['Generator', 'Date', 'MIA Method', 'Bandwidth'], axis=1).to_numpy(dtype=np.float32)

    error = abs(all_results2[:, 5] - all_results[:, 5]) * all_results[:, 5]
    y_error_pos = all_results[:, 5] + error
    y_error_neg = all_results[:, 5] - error

    plt.fill_between(all_results[:, 1], all_results[:, 5] - error, all_results[:, 5] + error, color=royal_blue,
                     alpha=0.5, edgecolor='#FFFFFF')

    plt.axhline(y=0.5, color='k', linestyle='--')

    plt.xlim([0, 40000])
    plt.ylim([0.39, 0.575])

    plt.title('AUC Against Amount of Reference Data Used',
                 fontname=font,
                 fontsize=title_size)

    plt.xlabel('Amount of Reference Data', fontname=font, fontsize=axes_size)
    plt.ylabel('AUC', fontname=font, fontsize=axes_size)

    # plt.xscale('log')

    plt.xticks(fontname=font, fontsize=axes_size)
    plt.yticks(fontname=font, fontsize=axes_size)

    plt.legend(labels=['AUC', 'Percentage Error', 'Baseline Accuracy'],
               fontsize=20)

    plt.show()


def plot_synth_error(all_results):
    plt.plot(all_results[:, 2], all_results[:, 5], color=dark_royal_blue)  # AUC

    all_results2 = pd.read_csv('Saved Data/train 1000, ref 1000, synth 5000, epochs 25/Synth Data Tests 2/synth_results2.csv').drop(
        ['Generator', 'Date', 'MIA Method', 'Bandwidth'], axis=1).to_numpy(dtype=np.float32)

    error = abs(all_results2[:, 5] - all_results[:, 5]) * all_results[:, 5]
    y_error_pos = all_results[:, 5] + error
    y_error_neg = all_results[:, 5] - error

    plt.fill_between(all_results[:, 2], all_results[:, 5] - error, all_results[:, 5] + error, color=royal_blue, alpha=0.5, edgecolor='#FFFFFF')

    plt.axhline(y=0.5, color='k', linestyle='--')

    plt.title('AUC Against Amount of Synthetic Data Produced',
                 fontname=font,
                 fontsize=title_size)

    plt.xlabel('Amount of Synthetic Data', fontname=font, fontsize=axes_size)
    plt.ylabel('AUC', fontname=font, fontsize=axes_size)

    plt.xlim([0, 40000])
    plt.ylim([0.49, 0.55])

    plt.xticks(fontname=font, fontsize=axes_size)
    plt.yticks(fontname=font, fontsize=axes_size)

    plt.legend(labels=['AUC', 'Percentage Error', 'Baseline Accuracy'],
               fontsize=20)

    plt.show()


if __name__ == "__main__":
    # Use here to plot data.
    OPTIONS = ['all_results', 'specific_results']
    SUB_OPTIONS = ['train', 'ref', 'synth', 'epochs', 'all']

    # For plotting all results
    FOLDER_NAME = "Epoch Tests"
    RESULTS = "epoch_results.csv"

    # For plotting specific results
    SPECIFIC_NAME = "TVAE_07-05-24-1244_1000_1000_5000_5_kde_mia_domias_evaluation_scott"  # Of specific graph
    information = ['TVAE', '1000', '1000', '5000', '5', 'Scott']  # TODO future make it take it auto from the text file

    TO_PLOT = OPTIONS[1]  # Choose what you want to plot
    INDEP_VAR = SUB_OPTIONS[3]

    if TO_PLOT == 'all_results':
        all_results = pd.read_csv(f'Saved Data/{FOLDER_NAME}/{RESULTS}').drop(['Generator', 'Date', 'MIA Method', 'Bandwidth'], axis=1).to_numpy(dtype=np.float32)

        if INDEP_VAR == 'train':
            plot_train_error(all_results)

        if INDEP_VAR == 'ref':
            plot_ref(all_results)

        if INDEP_VAR == 'synth':
            plot_synth_error(all_results)

        if INDEP_VAR == 'epochs':
            plot_epochs_error(all_results)

    elif TO_PLOT == 'specific_results':
        mem_set = pd.read_csv(f'Saved Data/{FOLDER_NAME}/{SPECIFIC_NAME}/mem_set.csv').drop(['Unnamed: 0'], axis=1).to_numpy(dtype=np.float32)
        reference_set = pd.read_csv(f'Saved Data/{FOLDER_NAME}/{SPECIFIC_NAME}/ref_set.csv').drop(['Unnamed: 0'], axis=1).to_numpy(dtype=np.float32)
        synth_set = pd.read_csv(f'Saved Data/{FOLDER_NAME}/{SPECIFIC_NAME}/synth_set.csv').drop(['Unnamed: 0'], axis=1).to_numpy(dtype=np.float32)
        x_test = pd.read_csv(f'Saved Data/{FOLDER_NAME}/{SPECIFIC_NAME}/x_test.csv').drop(['Unnamed: 0'], axis=1).to_numpy(dtype=np.float32)
        y_test = pd.read_csv(f'Saved Data/{FOLDER_NAME}/{SPECIFIC_NAME}/y_test.csv').drop(['Unnamed: 0'], axis=1).to_numpy(dtype=np.bool_).flatten()
        y_pred = pd.read_csv(f'Saved Data/{FOLDER_NAME}/{SPECIFIC_NAME}/y_pred.csv').drop(['Unnamed: 0'], axis=1).to_numpy(dtype=np.bool_).flatten()

        # plot_kde_difference(synth_set, reference_set, x_test, y_pred, information)  # UNFINISHED - IDK IF IS THE BEST ANYWAY
        plot_average_kde_vs_training_and_non_kde(synth_set, reference_set, x_test, y_pred, information)
        # plot_ecg_1(synth_set, reference_set, x_test, information)
