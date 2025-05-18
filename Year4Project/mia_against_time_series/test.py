from sklearn.metrics import accuracy_score, roc_auc_score

print('Calculating ACC and AUC...')
acc = accuracy_score([True, True, False, False], [True, False, False, False])
auc = roc_auc_score([True, True, False, False], [1, 1, 1, 1])

print('     ACC:', acc, '| AUC:', auc, '\n')