import pandas as pd
import numpy as np
from math import sqrt

imputed_data = pd.read_csv('MF_MCAR100_40.csv')
original = pd.read_csv('test.csv')

# original = original[100:200]

Y_imputed = imputed_data[['10']]
Y_original = original[['10']]

Y_imputed = Y_imputed.round()
print(Y_imputed)
print(Y_original)


Y_imputed = Y_imputed.to_numpy()
Y_original = Y_original.to_numpy()

res = Y_original == Y_imputed

print(res)

count = np.sum(res, axis=0)
print(count[0] / original.shape[0])
