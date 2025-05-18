# third party
import pandas as pd
from sdv.single_table import TVAESynthesizer as TVAE

# domias absolute
from domias.evaluator import evaluate_performance
from domias.models.generator import GeneratorInterface

import numpy as np
from scipy import signal
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler

# toby imports
import load_csv

array, metadata = load_csv.csv_to_numpy('./arrhythmia_dataset/all_records.csv')

print(array)
