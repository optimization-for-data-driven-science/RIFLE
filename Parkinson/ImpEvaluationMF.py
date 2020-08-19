import pandas as pd
import numpy as np
from math import sqrt

imputed_data = pd.read_csv('MF_Sydney_MCAR40_500_1.csv')
original = pd.read_csv('original_Sydney_MCAR40_500_1.csv')
missing = pd.read_csv('missing_Sydney_MCAR40_500_1.csv')

# y = imputed_data[['0']]
# print(y)
# exit(0)

rmse_list = []

features_names = list(original.columns)

for i in range(49):
    Y_imputed = imputed_data[[str(i)]]
    Y_original = original[[features_names[i]]]
    missing_Y = missing[[features_names[i]]]

    original_std = sqrt(Y_original.var()[0])
    # print(original_std)

    mask = missing_Y.isna()
    mask = mask.to_numpy()

    missing_entries = mask.sum(axis=0)[0]
    # print(missing_entries)

    Y_imputed = Y_imputed.to_numpy()
    Y_original = Y_original.to_numpy()

    mse = np.linalg.norm(np.multiply(Y_imputed - Y_original, mask)) ** 2 / missing_entries
    print("Feature name: ", features_names[i])
    print("RMSE: ", sqrt(mse))
    print("Scaled: ", sqrt(mse) / original_std)
    print("-----------------------------------")
    rmse_list.append(sqrt(mse) / original_std)

print(rmse_list)

for item in rmse_list:
    print(item)
