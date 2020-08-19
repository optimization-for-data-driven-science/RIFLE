from math import sqrt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

"""
data = pd.read_csv("Synthetic_missing_MNAR_P40_2000.csv")

print(data)

X = data[data.columns[0:-1]]
Y = data[['200']]

X = X.to_numpy()
Y = Y.to_numpy()
"""

X = np.loadtxt('MCARBeta_P80_X2000.csv', delimiter=',')

Y = np.loadtxt('MCARBeta_P80_Y2000.csv', delimiter=',')
Y = Y[:, np.newaxis]

data_points = X.shape[0]

mask_X = np.isnan(X)
mask_X = np.ones(shape=mask_X.shape) - mask_X
print(mask_X)

train_std = np.nanstd(Y)

mask_Y_test = np.isnan(Y)
missing_entries = mask_Y_test.sum(axis=0)[0]
mask_Y = np.ones(shape=mask_Y_test.shape) - mask_Y_test
print(missing_entries)
print(mask_Y.shape)

sc = StandardScaler()
sc_y = StandardScaler()

sc.fit(X)
sc_y.fit(Y)

X = sc.transform(X)
Y = sc_y.transform(Y)

X[np.isnan(X)] = 0
Y[np.isnan(Y)] = 0

C = np.dot(X.T, X) / np.dot(mask_X.T, mask_X)
b = np.dot(X.T, Y) / np.dot(mask_X.T, mask_Y)

# 1500000
lam = 2

predicts = []
for i in range(len(mask_X)):
    row_i = mask_X[i]
    nonzeros = np.nonzero(row_i)[0]
    nonzeros = list(nonzeros)

    currentC = C[nonzeros, :]
    currentC = currentC[:, nonzeros]

    currentB = b[nonzeros, :]
    theta = np.dot(np.linalg.inv(currentC + lam * np.identity(currentC.shape[0])), currentB)

    data_i = X[i]
    data_i = data_i[:, np.newaxis]
    data_i = data_i[nonzeros, :]

    y_predict = np.dot(data_i.T, theta)
    predicts.append(y_predict[0][0])

Y_pred = np.array(predicts)
Y_predictions = Y_pred[:, np.newaxis]
Y_predictions = Y_predictions * sqrt(sc_y.var_[0]) + sc_y.mean_[0]


Y_original = np.loadtxt('Beta_Y2000.csv', delimiter=',')
Y_original = Y_original[:, np.newaxis]

original_std = np.nanstd(Y_original)

mse = np.linalg.norm(np.multiply(Y_predictions - Y_original, mask_Y_test)) ** 2 / missing_entries
print("RMSE: ", sqrt(mse))
print("Scaled: ", sqrt(mse) / original_std)
