import pandas as pd
import numpy as np
from math import sqrt

imputed_data = pd.read_csv('MF_BC60.csv')
original = pd.read_csv('BC.csv')
missing = pd.read_csv('missing_BC_MCAR60.csv')

feature_names = []

rmse_list = []
for i in range(1, 34):
    feature_names.append(str(i))

for feature_name in feature_names:
    Y_imputed = imputed_data[[feature_name]]
    Y_original = original[[feature_name]]
    missing_Y = missing[[feature_name]]

    original_std = sqrt(Y_original.var()[0])
    # print(original_std)

    mask = missing_Y.isna()
    mask = mask.to_numpy()

    missing_entries = mask.sum(axis=0)[0]
    # print(missing_entries)

    Y_imputed = Y_imputed.to_numpy()
    Y_original = Y_original.to_numpy()

    mse = np.linalg.norm(np.multiply(Y_imputed - Y_original, mask)) ** 2 / missing_entries
    print("Feature name: ", feature_name)
    print("RMSE: ", sqrt(mse))
    print("Scaled: ", sqrt(mse) / original_std)
    print("-----------------------------------")
    rmse_list.append(sqrt(mse) / original_std)

print(rmse_list)

for item in rmse_list:
    print(item)
