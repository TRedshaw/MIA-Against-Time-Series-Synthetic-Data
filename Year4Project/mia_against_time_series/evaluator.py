# Third Party
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.neighbors import KernelDensity


def kde_mia_domias_evaluation(synth_set, reference_set, x_test, y_test, bandwidth='scott'):
    """KDE Estimation"""
    density_gen = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(synth_set)
    density_data = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(reference_set)

    '''KDE Evaluation'''
    # Returns a list of log-likelihood that each x_test sample/row is in the synthetic/reference dataset.
    # Higher values (closer to 0) indicate a better model, as log(1) will be closer to 0, and log(0) is infinite.
    # Makes the PDF of each x_test and compares to the model you're calling the object of.

    p_G_evaluated = density_gen.score_samples(x_test)
    p_R_evaluated = density_data.score_samples(x_test)
    p_rel = p_R_evaluated/p_G_evaluated

    # TESTING PRINTS
    # print(p_G_evaluated)
    # print(p_R_evaluated)
    # print('unlogged\n', np.exp(p_R_evaluated))
    # print(np.exp(p_R_evaluated))

    # From DOMIAS evaluator.py but flipped - their eqn. hence mention in report
    # Therefore, the ratio needs to be reference/synthetic evaluation, where a ratio of larger than 1 will
    # indicate better fit to synthetic data

    '''MIA Success'''  # From DOMIAS baselines.py
    y_pred = p_rel > np.median(p_rel)

    # Check if equivalent to DOMIAS
    # y_pred_domias = np.exp(p_G_evaluated)/np.exp(p_R_evaluated)>np.median(np.exp(p_G_evaluated)/np.exp(p_R_evaluated))
    # print(y_pred, y_pred_domias)
    # if y_pred_domias.all() == y_pred.all():
    #     print('Hypothesis of log version of DOMIAS is correct')

    print('Calculating ACC and AUC...')
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, p_rel)

    print('     ACC:', acc, '| AUC:', auc, '\n')

    return acc, auc, y_pred


def time_sensitive_kde():
    """See my Notion for 'my method'. """
    pass


def kde_mia_domias_and_euclidian():
    pass
