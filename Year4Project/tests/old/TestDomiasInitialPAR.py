#
#
#
# Used to send sizes of data, and sent it to the DOMIAS code, which then returned the synth set along with the
# split data e.g. training and reference. This was then saved into files in folders. SPECIFICALLY USING PAR.
# Was originally meant to return the performance, however couldn't get the KDE in DOMIAS to work
#
#
#


# third party
import pandas as pd
from sdv.single_table import TVAESynthesizer as TVAE

# domias absolute
from domias.evaluator import evaluate_performance
from domias.models.generator import GeneratorInterface

import numpy as np
from scipy import signal, stats
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler

# toby imports
import load_csv
import time
from sklearn.neighbors import KernelDensity
from sdv.sequential import PARSynthesizer as PAR
import seaborn as sns
import matplotlib.pyplot as plt


def get_generator(metadata, epochs: int = 1000, seed: int = 0) -> GeneratorInterface:
    class LocalGenerator(GeneratorInterface):
        def __init__(self) -> None:
            self.model = PAR(epochs=epochs, metadata=metadata)

        def fit(self, data: pd.DataFrame) -> "LocalGenerator":
            self.model.fit(data)
            return self

        def generate(self, count: int) -> pd.DataFrame:
            return self.model.sample(count)

    return LocalGenerator()


path = '../../mia_against_time_series/arrhythmia_dataset/all_records.csv'

# Loading the dataset
dataset, all_records_metadata = load_csv.csv_to_numpy(file=path, downsample=False)

print('full dataset shape', dataset.shape)
# print(all_records_metadata)

# print(all_records_metadata[:, ::3].shape)

# def get_dataset() -> np.ndarray:
#     def data_loader() -> np.ndarray:
#         scaler = StandardScaler()
#         np.random.shuffle(data)
#         return scaler.fit_transform(data)
#
#     return data_loader()
#
#
# dataset = get_dataset()


# Downsample the data
# dataset = dataset[:, 0::3]
# print(all_records_metadata)


#  80/20 split training -> testing
dataset_80 = dataset[:int(0.8 * len(dataset))]
dataset_20 = dataset[int(0.8 * len(dataset)):]

print('train set shape', dataset_80.shape, 'test set shape', (dataset_20.shape))
# print(len(dataset_80)//10, len(dataset_20)//10)


# Mem_set = training set
mem_set_size = len(dataset_80)//10  # divide by 10 to reduce amount by a lot as currently a lot
reference_set_size = len(dataset_20)//10
training_epochs = 10
synthetic_sizes = [len(dataset_20)//10]  # how many samples to test the attacks on
density_estimator = "kde"  # prior, kde, bnaf

generator = get_generator(
    epochs=training_epochs,
    metadata=all_records_metadata
)


mem_set, reference_set, synth_set, X_test, Y_test = evaluate_performance(
    generator,
    dataset,
    mem_set_size,
    reference_set_size,
    training_epochs=training_epochs,
    synthetic_sizes=synthetic_sizes,
    density_estimator=density_estimator,
)

SAVE_VARS_TO_FILES = False
VARS_PATH = '/Users/toby/PycharmProjects/Year4Project/Year4Project/mia_against_time_series/Variables 24-04-16 2026'

if SAVE_VARS_TO_FILES:
    pd.DataFrame(mem_set).to_csv("Variables 24-04-16 2026/mem_set.csv")
    pd.DataFrame(reference_set).to_csv("Variables 24-04-16 2026/ref_set.csv")
    pd.DataFrame(synth_set).to_csv("Variables 24-04-16 2026/synthetic_data.csv")
    pd.DataFrame(X_test).to_csv("Variables 24-04-16 2026/x_test.csv")
    pd.DataFrame(Y_test).to_csv("Variables 24-04-16 2026/y_test.csv")



# sns.kdeplot(reference_data[0:3, :].transpose(1, 0), shade=True)
# sns.kdeplot(synthetic_data.values[0:3, :].transpose(1, 0))
# plt.show()





# assert 100 in perf
# results = perf[100]
#
# assert "MIA_performance" in results
# assert "MIA_scores" in results
#
# print(results["MIA_performance"])