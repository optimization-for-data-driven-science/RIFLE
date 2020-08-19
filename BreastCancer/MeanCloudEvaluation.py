import pandas as pd
import numpy as np
from math import sqrt

feature_names = []

ind = 10
rmse_list = []
for i in range(0, 48):
    feature_names.append(str(i))

rmses = []

# original = pd.read_csv('original_cloud_MCAR70_100_' + str(ind) + '.csv')
# missing = pd.read_csv('cloud_MCAR70_100_' + str(ind) + '.csv')

original = pd.read_csv('original_drive_MCAR80_2000.csv')
missing = pd.read_csv('drive_MCAR80_2000.csv')

imputed = missing.copy()
imputed = imputed.fillna(missing.mean())

for feature_name in feature_names:

    Y_imputed = imputed[[feature_name]]
    # Y_imputed = imputed_data[['X' + feature_name]]
    Y_original = original[[feature_name]]
    missing_Y = missing[[feature_name]]

    original_std = sqrt(Y_original.var()[0])

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
