import pandas as pd
import numpy as np
from math import sqrt

imputed_data = pd.read_csv('final_MCAR_Amelia80n1000_11.csv')
original = pd.read_csv('music_original_MCAR_P80_1000_1.csv')
missing = pd.read_csv('music_missing_MCAR_P80_1000_1.csv')

Y_imputed = imputed_data[['X0']]
Y_original = original[['0']]
missing_Y = missing[['0']]

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
