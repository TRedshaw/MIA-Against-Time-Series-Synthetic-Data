#
#
#
# I don't need this anymore as it involved a lot of the KDE production and MIA calculations and i now have them in code
# as well as the plots
#
#
#


from sklearn.metrics import accuracy_score, roc_auc_score

import load_csv
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats, linalg
import CustomGaussiaKde
from sklearn.neighbors import KernelDensity

# TVAE 500 epochs 24-04-17 0913
# TVAE 10 epochs 24-04-16 1829

# Load global variables
mem_set = pd.read_csv('../../mia_against_time_series/TVAE 500 epochs 24-04-17 0913/mem_set.csv').drop(['Unnamed: 0'], axis=1).to_numpy(dtype=np.float32)
ref_set = pd.read_csv('../../mia_against_time_series/TVAE 500 epochs 24-04-17 0913/ref_set.csv').drop(['Unnamed: 0'], axis=1).to_numpy(dtype=np.float32)
synth_set = pd.read_csv('../../mia_against_time_series/TVAE 500 epochs 24-04-17 0913/synth_set.csv').drop(['Unnamed: 0'], axis=1).to_numpy(dtype=np.float32)
x_test = pd.read_csv('../../mia_against_time_series/TVAE 500 epochs 24-04-17 0913/x_test.csv').drop(['Unnamed: 0'], axis=1).to_numpy(dtype=np.float32)
y_test = pd.read_csv('../../mia_against_time_series/TVAE 500 epochs 24-04-17 0913/y_test.csv').drop(['Unnamed: 0'], axis=1).to_numpy(dtype=np.float32)

# Create custom variables
avg_ref_set = np.mean(ref_set, axis=0)
avg_synth_set = np.mean(synth_set, axis=0)


def plot_kde_basic():
    sns.kdeplot(ref_set[0:3, :].transpose(1, 0), fill=True)
    sns.kdeplot(synth_set[0:3, :].transpose(1, 0))
    plt.show()


def plot_kde_avg():
    sns.kdeplot(avg_ref_set, fill=True)
    sns.kdeplot(avg_synth_set)
    plt.show()


# @@@ SVD so can use proper gaussian - do i then make x_test using this
# U, s, VT = linalg.svd(synthetic_data)
# print(U)


# @@@ Scikit KDE
print('synth set shape', synth_set.shape)
print('mem (training) set shape', mem_set.shape)
print('ref (population) set shape', ref_set.shape)
print('x_test (synth gen members and non members) shape', x_test.shape)

# DON'T HAVE TO DO THIS - IF I DO: 72% VS 38% ATTACK SUCCESS? WHY
# mem_set_small_of_x_test = x_test[0:950, :]
# ref_set_small_of_x_test = x_test[7607:7607+951, :]
# new_x_test = np.vstack((mem_set_small_of_x_test, ref_set_small_of_x_test))
# print(new_x_test.shape)
#
# mem_set_small_of_y_test = y_test[0:950, :]
# ref_set_small_of_y_test = y_test[7607:7607+951, :]
# new_y_test = np.vstack((mem_set_small_of_y_test, ref_set_small_of_y_test))
# print(new_y_test.shape)

density_gen = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(synth_set[:, :])
density_data = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(ref_set)

'''Evaluating properly'''
# TEST PASSING IN THE EXACT SAME SETS COZ THEN SHOULD GET 100% OR 0% IF I PASS IN NOTHING
# I don't have to pass in the same size everything????
p_G_evaluated = density_gen.score_samples(x_test)
p_R_evaluated = density_data.score_samples(x_test)

p_rel = p_G_evaluated/p_R_evaluated
y_pred = p_rel > np.median(p_rel)
acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, p_rel)
print(acc, auc)


'''Evaluating with themselves, to test correct function'''


test_x_test = x_test[0:1901]
test_y_test = y_test[0:1901]

p_G_evaluated = density_gen.score_samples(test_x_test)
p_R_evaluated = density_data.score_samples(test_x_test)

p_rel = p_G_evaluated/p_R_evaluated
print(p_rel)

y_pred = p_rel >= np.median(p_rel)
print(y_pred)
acc = accuracy_score(test_y_test, y_pred)
auc = roc_auc_score(test_y_test, p_rel)

print(acc, auc)

# @@@ SciPY KDE
# density_gen = CustomGaussiaKde.GaussianKde(synthetic_data.transpose(1, 0))
# density_data = CustomGaussiaKde.GaussianKde(ref_set.transpose(1, 0))

# plot_kde_basic()

# plot_kde_avg()

plt.plot(x_test[0])
plt.plot(synth_set[0])
plt.plot(ref_set[0])
plt.show()

# print(density_data(ref_set.T))

# p_R_evaluated = density_data(np.linspace(x_test.min(), x_test.max(), 488).T)
# print(p_R_evaluated)
