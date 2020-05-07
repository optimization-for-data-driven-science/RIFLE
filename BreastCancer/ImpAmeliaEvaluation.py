import pandas as pd
import numpy as np
from math import sqrt

feature_names = []

rmse_list = []
for i in range(1, 34):
    feature_names.append(str(i))

rmses = []
for feature_name in feature_names:
    imputed_data = pd.read_csv('MICE_BC_MCAR10.csv')
    original = pd.read_csv('BC.csv')
    missing = pd.read_csv('missing_BC_MCAR10.csv')

    Y_imputed = imputed_data[['X' + feature_name]]
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
    print("RMSE: ", sqrt(mse))
    print("Scaled: ", sqrt(mse) / original_std)
    rmses.append(sqrt(mse) / original_std)
    print("--------------------------------")
for item in rmses:
    print(item)
