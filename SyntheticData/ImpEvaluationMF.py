import pandas as pd
import numpy as np
from math import sqrt

imputed_data = pd.read_csv('mf_sparse_data_mcar40.csv')

Y_original = np.loadtxt('Sparse_Y2000.csv', delimiter=',')
Y_original = Y_original[:, np.newaxis]

missing_data = pd.read_csv('sparse_data_mcar40.csv')
Y_missing = missing_data[['200']]
Y_missing = Y_missing.to_numpy()
# Y_missing = np.loadtxt('sparse_data_mcar40.csv', delimiter=',')
# Y_missing = Y_missing[:, np.newaxis]

Y_imputed = imputed_data[['200']]

original_std = np.nanstd(Y_original)
print(original_std)

mask = np.isnan(Y_missing) + np.zeros(shape=Y_missing.shape)
missing_entries = mask.sum(axis=0)[0]
print(missing_entries)
Y_imputed = Y_imputed.to_numpy()

mse = np.linalg.norm(np.multiply(Y_imputed - Y_original, mask)) ** 2 / missing_entries
print("RMSE: ", sqrt(mse))
print("Scaled: ", sqrt(mse) / original_std)


