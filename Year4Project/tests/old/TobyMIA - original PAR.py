#
#
#
# this was the first edition of using PARsynthesiser, however needed to be made a lot better with data handling etc.
# so it was more similar to the TVAE one so could become modular, also some methods were incorrect
#
#
#

from sdv.metadata import SingleTableMetadata
from sdv.sequential import PARSynthesizer
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.neighbors import KernelDensity

import load_csv
import numpy as np
import pandas as pd
import torch
from scipy import stats
from scipy.stats import multivariate_normal
import seaborn as sns
import matplotlib.pyplot as plt
# from sklearn.neighbors import KernelDensity

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

path = '../../mia_against_time_series/arrhythmia_dataset/all_records.csv'

'''OLD'''
# # Loading the dataset
# metadata = SingleTableMetadata()
# dataset_df = pd.read_csv(path).drop(['Unnamed: 0'], axis=1)
# dataset_df = dataset_df.drop('Class', axis=1)
# dataset_df = dataset_df.drop('Patient', axis=1)
# dataset_df = dataset_df.transpose()
# dataset_df.insert(0, "time_point", range(1, 489))
# dataset_df.columns = dataset_df.columns.astype(str)
# # print(dataset_df)
#
# metadata.detect_from_dataframe(dataset_df)
# metadata.update_column(column_name='time_point', sdtype='id')
# metadata.set_sequence_key(column_name='time_point')
# # print(metadata)
#
# # quit()

'''Load Data and setup for PARSynth'''
# Load data
print('Loading dataset...')
metadata = SingleTableMetadata()
dataset_df = pd.read_csv(path).drop(['Unnamed: 0'], axis=1)
dataset_df = dataset_df.drop('Class', axis=1)
dataset_df = dataset_df.drop('Patient', axis=1)
dataset_size = len(dataset_df)

# plt.plot(dataset_df.to_numpy()[0])
# plt.show()

# Deconstruct data and create table keys
dataset = dataset_df.to_numpy().flatten().T
ecg_id = np.repeat(np.arange(95082)+1, 488).T
ecg_sequence = np.tile(np.arange(488)+1, 95082).T

# Setup correct table structure for PARsynthesiser
print('Restructuring dataset...')
par_synth_dataset_df = pd.DataFrame()
par_synth_dataset_df.insert(0, "ecg_id", ecg_id)
par_synth_dataset_df.insert(1, "ecg_seq", ecg_sequence)
par_synth_dataset_df.insert(2, "ecg_val", dataset)
par_synth_dataset_df.columns = par_synth_dataset_df.columns.astype(str)

# Get metadata
print('Getting metadata...')
metadata.detect_from_dataframe(par_synth_dataset_df)
metadata.update_column(column_name='ecg_id', sdtype='id')
metadata.set_sequence_key(column_name='ecg_id')
metadata.set_sequence_index(column_name='ecg_seq')

print(par_synth_dataset_df)

''' HYPERPARAMETERS
        Setting up data sizes for generation.
    
    Variables:
        mem_set_size: int
            Amount of training data for synthetic data generation
        reference_set_size: int
            Amount of data from dataset used as the 'population' dataset
        training_epochs: int
            Number of epochs for synthetic data generation
        synthetic_sizes: int
            Number of synthetic ECG signals to produce
'''
print('Setting hyperparameters...')
# From the dataset of 95082 - or dataset_size
mem_set_size = 20  # round(0.4*dataset_size)  # ensure small enough s.t 2*mem_set does not overlap with dataset_size-ref_Set_size
reference_set_size = round(0.02*dataset_size)
training_epochs = 1
synthetic_sizes = 1  # how many synthetic samplers to create




'''Setting up data sizes from hyperparameters'''
print('Creating hyperparameter variables...')
# TODO Make the 488 work for any size data
# Make and keep mem set as a vertical by 3 dataframe for the generation
mem_set = par_synth_dataset_df[:mem_set_size*488]  # Make training set the size we dictated - must be df

# Ref and non_mem must be np arrays for KDE etc.
ref_set = par_synth_dataset_df[-reference_set_size*488:].to_numpy()[:, 2]
non_mem_set = par_synth_dataset_df[mem_set_size*488:2*488*mem_set_size].to_numpy()[:, 2]
'''Setup set of training members and non-members - Y_test is flags for if members or not'''
# get real test sets of members and non-members
x_test = np.vstack((mem_set.to_numpy(), non_mem_set))
y_test = np.concatenate([np.ones(len(mem_set)//488), np.zeros(len(non_mem_set)//488)]).astype(bool)


'''Generating synthetic data'''
print('Generating synthetic data...')
synthesizer = PARSynthesizer(metadata=metadata, epochs=training_epochs, verbose=True)
synthesizer.fit(mem_set)  # Training the synthesizer with the dataset
synth_set = synthesizer.sample(num_sequences=synthetic_sizes)  # Generating the synthetic dataset


'''Reshaping dataframes for KDE'''
print('Reshaping data for KDE...')
synth_set_reshape = np.reshape(synth_set[:, 2], (488, synthetic_sizes))
ref_set_reshape = np.reshape(ref_set[:, 2], (488, reference_set_size))
x_test_reshape = np.reshape(x_test[:, 2], (488, mem_set_size*2))

plt.plot(mem_set.to_numpy()[0:488, 2])
plt.plot(ref_set_reshape[0, :])
plt.plot(ref_set_reshape[0, :])
plt.show()

print(synth_set_reshape.shape)
print(ref_set_reshape.shape)
print(x_test_reshape.shape)
print(x_test_reshape)
print(y_test)



# y test is the flag array for if values are members or not





'''KDE'''
print('Constructing PDF using KDE for synthetic data and reference set...')
density_gen = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(synth_set_reshape)
density_data = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(ref_set_reshape)

'''Evaluating properly'''
# TEST PASSING IN THE EXACT SAME SETS COZ THEN SHOULD GET 100% OR 0% IF I PASS IN NOTHING
# I don't have to pass in the same size everything????

print('Evaluating x_test...')
p_G_evaluated = density_gen.score_samples(x_test_reshape)
p_R_evaluated = density_data.score_samples(x_test_reshape)

print('Using DOMIAS score Eqn...')
p_rel = p_G_evaluated/p_R_evaluated

print('Deduce preducted members above median p_rel value...')
y_pred = p_rel > np.median(p_rel)

print('Calculating ACC and AUC...')
acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, p_rel)

print('ACC:', acc, '| AUC:', auc)

# '''Print statements'''
# print('synth set shape', synthetic_data.values.shape)
# print('mem (training) set shape', mem_set.shape)
# print('ref (population) set shape', reference_data.shape)
# print('x_test (synth gen members and non members) shape', X_test.shape)
# 
# print('synth set', synthetic_data.values[0, (0, len(synthetic_data.values[0])-1)])
# print('mem set', mem_set[0, (0, len(mem_set[0])-1)])
# print('ref set', reference_data[0, (0, len(reference_data[0])-1)])
# print('x test', X_test[0, (0, len(X_test[0])-1)])

'''Plots'''
# # Plotting ECG Signals - first in each array
# plt.plot(x_test_reshape[0])
# plt.plot(synth_set_reshape[0])
# plt.plot(ref_set_reshape[0])
# plt.show()
#
# # Plotting KDE of synth and ref set (synth = filled)
# sns.kdeplot(synth_set_reshape.transpose(1, 0), fill=True)
# sns.kdeplot(ref_set_reshape.transpose(1, 0))
# plt.show()
