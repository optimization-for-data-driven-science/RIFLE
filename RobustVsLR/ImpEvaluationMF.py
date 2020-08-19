import pandas as pd
import numpy as np
from math import sqrt

imputed_data = pd.read_csv('MF_Beta_MCAR_85_10_1000.csv')
original = pd.read_csv('original_Beta_MCAR_85_10_1000.csv')
missing = pd.read_csv('missing_Beta_MCAR_85_10_1000.csv')

Y_imputed = imputed_data[['78']]
Y_original = original[['critical_temp']]
missing_Y = missing[['critical_temp']]

original_std = sqrt(Y_original.var()[0])
print(original_std)

mask = missing_Y.isna()
mask = mask.to_numpy()

missing_entries = mask.sum(axis=0)[0]
print(missing_entries)

Y_imputed = Y_imputed.to_numpy()
Y_original = Y_original.to_numpy()

mse = np.linalg.norm(np.multiply(Y_imputed - Y_original, mask)) ** 2 / missing_entries
print("RMSE: ", sqrt(mse))
print("Scaled: ", sqrt(mse) / original_std)
