# Custom modules
import load_csv
import generator
import evaluator
import plotter
import save_test_data

# Third party modules
import warnings
import torch
import numpy as np
from datetime import datetime
import os


# Ignore warnings
warnings.filterwarnings("ignore")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

date = datetime.now().strftime("%d-%m-%y-%H%M")  # Time will be as batch - put into loop if want each iteration

mia_types = ['kde_mia_domias_evaluation', 'time sensitive_kde', 'kde_mia_domias_and_euclidian']


'''Load Data'''
path = './arrhythmia_dataset/all_records.csv'  # Input data CSV to load (each row is a time signal)

dataset, all_records_metadata = load_csv.csv_to_numpy(file=path, downsample=False)
dataset_size = len(dataset)
signal_length = dataset.shape[1]

''' HYPERPARAMETERS
        Setting up data sizes for generation.

    Variables:
        mem_set_sizes: List[int]
            Training data sizes to loop through.
        reference_set_sizes: List[int]
            Amount of data from dataset used as the 'population' dataset
        synthetic_sizes: List[int]
            Number of synthetic ECG signals to produce
        training_epochs: List[int]
            Number of epochs for synthetic data generation
        generation_method: str
            'TVAE' or 'PAR' for the synthesiser option
        bandwidth: str / int
            str: {'scott' or 'silverman'} 
            int: 0 -> 1
        mia_method: list selection
            Choose from the set "mia_type" - Currently only 0 works
        plot_whilst_running: bool
            True = Plot graphs each run (will stop code progressing)
        save_data: bool
            True = Save data to 'Saved Data' file

'''

# Set hyperparameters for testing
mem_set_sizes = [1000]  # 1000 ; ensure small enough s.t 2*mem_set does not overlap with dataset_size-ref_set_size
reference_set_sizes = [1000]  # 1000
synthetic_sizes = [1]  # 5000
training_epochs = [25]  # 50 / 25
generation_method = 'PAR'  # "TVAE" or "PAR"
bandwidth = 'scott'
mia_method = mia_types[0]
plot_whilst_running = False
save_data = True


# Loop through varied training set sizes
for mem_set_size in mem_set_sizes:  # 10000
    for reference_set_size in reference_set_sizes:  # 10000
        for training_epoch in training_epochs:  # 100
            for synthetic_size in synthetic_sizes:  # 10000
                print(f"Parameters:\n"
                      f"    mem set size (no. ecgs) = {mem_set_size}\n"
                      f"    ref set size (no. ecgs) = {reference_set_size}\n"
                      f"    synth set size (no. ecgs) = {synthetic_size}\n"
                      f"    number of epochs = {training_epoch}\n"
                      f"    generation method = {generation_method}\n"
                      f"    bandwidth = {bandwidth}\n")

                # print('Creating data subsets...\n')
                '''Generate data subsets'''
                mem_set = dataset[:mem_set_size]  # Make training set the size we dictated - mem_set becomes df for generation
                # Ref and non_mem must STAY AS np arrays for KDE etc.
                non_mem_set = dataset[mem_set_size:2*mem_set_size]
                reference_set = dataset[-reference_set_size:]  # Take reference (test) set from end of dataset
                x_test = np.concatenate([mem_set, non_mem_set])  # Can do this as here mem_set is still np and proper table
                y_test = np.concatenate([np.ones(mem_set.shape[0]), np.zeros(non_mem_set.shape[0])]).astype(bool)

                '''Generate data'''
                if generation_method == 'PAR':
                    synth_set = generator.par_time_synthesiser(mem_set, mem_set_size, synthetic_size, signal_length, training_epoch)
                if generation_method == 'TVAE':
                    synth_set = generator.tvae_time_synthesiser(mem_set, synthetic_size, training_epoch, all_records_metadata)

                '''Evaluate'''
                # TODO Will need to add other methods ifs, if I create them
                if mia_method == 'kde_mia_domias_evaluation':
                    acc, auc, y_pred = evaluator.kde_mia_domias_evaluation(synth_set, reference_set, x_test, y_test, bandwidth)

                '''Save Data'''
                if save_data:
                    save_test_data.update_all_test_results(generation_method, date, mem_set_size, reference_set_size,
                                                           synthetic_size, training_epoch, mia_method, bandwidth,
                                                           acc, auc)

                    folder_name = f'{generation_method}_{date}_{mem_set_size}_{reference_set_size}_{synthetic_size}_{training_epoch}_{mia_method}_{bandwidth}'
                    os.mkdir(f"Saved Data/{folder_name}")

                    save_test_data.save_matrices(folder_name, mem_set, reference_set, synth_set, x_test, y_test, y_pred)

                    save_test_data.save_info(folder_name, generation_method, date, mem_set_size, reference_set_size,
                                             synthetic_size, training_epoch, mia_method, bandwidth, acc, auc)

                # Plot
                if plot_whilst_running:
                    information = [generation_method, mem_set_size, reference_set_size, synthetic_size, training_epoch, bandwidth]
                    plotter.plot_ecg_1(synth_set, reference_set, x_test, information)
                    plotter.plot_average_kde_vs_training_and_non_kde(synth_set, reference_set, x_test, y_pred, information)
