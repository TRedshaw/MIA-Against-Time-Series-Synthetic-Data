# ECG dataset for training
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset, WeightedRandomSampler
from torch.utils.data import random_split
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

import numpy as np
from scipy import signal
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm
from scipy.stats import kde

import sys
import os
import math
import time
import json
import pickle
import copy
import random
from copy import deepcopy
from tqdm.auto import tqdm
from datetime import datetime as dt

import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import seaborn as sns
from IPython.display import display

def load_ecg_data(path="./arrhythmia_dataset/all_records.csv", np=True, drop_class=True, drop_patient=True):
    data = pd.read_csv(path, index_col=0).reset_index(drop=True)
    class_data = data['Class']
    patient_data = data['Patient']
    cols = set(data.columns)
    cols.remove('Class')
    cols.remove('Patient')
    if drop_class:
        data = data.drop('Class', axis=1)
    if drop_patient:
        data = data.drop('Patient', axis=1)
    # y = data.values.astype(float)
    y = data.loc[:, ~data.columns.isin(["Class", "Patient"])].values.astype(float)
    # y = data.loc[:, data.columns != 'Class' ].values.astype(float)
    f = signal.decimate(y, 3)
    data = pd.DataFrame(f)

    if np:
        return data.to_numpy()
    elif not drop_class and not drop_patient:
        data = pd.concat([data, class_data], axis=1)
        return pd.concat([data, patient_data], axis=1)
    elif not drop_class:
        return pd.concat([data, class_data], axis=1)
    elif not drop_patient:
        return pd.concat([data, patient_data], axis=1)
    else:
        return data

print(torch.__version__)

if torch.cuda.is_available():
  device = torch.device("cuda")
else:
  device = torch.device("cpu")


class ECGDataset(Dataset):
    def __init__(self, file_loc="arrhythmia_dataset/all_records.csv"):
        data = load_ecg_data(file_loc)
        data = np.expand_dims(data, axis=(2))

        max_sequence_length = data.shape[1]
        sequence_lengths_list = [max_sequence_length] * len(data)

        self.data = np.array(data)
        self.sequence_lengths_list = sequence_lengths_list
        self.features_per_timestep = data.shape[2]
        self.max_sequence_length = max_sequence_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.sequence_lengths_list[idx]

## THIS CODE CREATES THE TRAINING AND TEST DATASET
# file_loc = "./arrhythmia_dataset/all_records.csv"
# data = pd.read_csv(file_loc, index_col=0)
# # Patients 201 and 202 are the same person
# data.loc[data['Patient']==202, 'Patient']=201
# data.reset_index(inplace=True, drop=True)
# # shuffle the data and split into train and validation sets
# X_train, X_test, _, _ = train_test_split(data, pd.Series([1]*len(data)), test_size=0.2, shuffle=True, random_state=2, stratify=data['Class'])
# # save to files
# X_train.to_csv('all_records_train.csv')
# X_test.to_csv('all_records_test.csv')
# # Check same proportions in each dataset
# X_train['Class'].value_counts(normalize=True).to_frame()
# X_test['Class'].value_counts(normalize=True).to_frame()

file_loc = "../../mia_against_time_series/arrhythmia_dataset/all_records.csv"
# data = pd.read_csv(file_loc, index_col=0)
# data.reset_index(inplace=True, drop=True)
# print(data)

ECGData = ECGDataset()
print(ECGData.data)