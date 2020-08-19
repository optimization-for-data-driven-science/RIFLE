import pandas as pd
import numpy as np
from math import sqrt

imputed_data = pd.read_csv('MF_Spam_MCAR30.csv')
original = pd.read_csv('spam.csv')
missing = pd.read_csv('missing_Spam_MCAR30.csv')

feature_names = []

original_columns = list(original.columns)
rmse_list = []
for i in range(0, 57):
    feature_names.append(str(i))

for i in range(len(original_columns)):
    Y_imputed = imputed_data[[str(i)]]
    Y_original = original[[original_columns[i]]]
    missing_Y = missing[[original_columns[i]]]

    original_std = sqrt(Y_original.var()[0])
    # print(original_std)

    mask = missing_Y.isna()
    mask = mask.to_numpy()

    missing_entries = mask.sum(axis=0)[0]
    # print(missing_entries)

    Y_imputed = Y_imputed.to_numpy()
    Y_original = Y_original.to_numpy()

    mse = np.linalg.norm(np.multiply(Y_imputed - Y_original, mask)) ** 2 / missing_entries
    print("Feature name: ", original_columns[i])
    print("RMSE: ", sqrt(mse))
    print("Scaled: ", sqrt(mse) / original_std)
    print("-----------------------------------")
    rmse_list.append(sqrt(mse) / original_std)

print(rmse_list)

for item in rmse_list:
    print(item)
