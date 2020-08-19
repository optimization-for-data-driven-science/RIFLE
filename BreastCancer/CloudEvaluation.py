import pandas as pd
import numpy as np
from math import sqrt

feature_names = []

ind = 400
rmse_list = []
for i in range(0, 48):
    feature_names.append(str(i))

rmses = []

num = 1

for feature_name in feature_names:
    imputed_data = pd.read_csv('MICE_drive_MCAR80_3000.csv')
    # imputed_data = pd.read_csv('MF_drive_MCAR30_' + str(ind) + '_' + str(num) + '.csv')
    imputed_data = imputed_data.drop(['Unnamed: 0'], axis=1)

    # original = pd.read_csv('original_drive_MCAR30_' + str(ind) + '_' + str(num) + '.csv')
    # missing = pd.read_csv('drive_MCAR30_' + str(ind) + '_' + str(num) + '.csv')
    original = pd.read_csv('original_drive_MCAR80_3000.csv')
    missing = pd.read_csv('drive_MCAR80_3000.csv')

    feature_columns = list(imputed_data.columns)
    if 'X' + feature_name not in feature_columns:
        continue

    # Y_imputed = imputed_data[[feature_name]]
    Y_imputed = imputed_data[['X' + feature_name]]
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
