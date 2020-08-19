import pandas as pd
import numpy as np
import glob
from math import sqrt
from sklearn.preprocessing import StandardScaler


data = pd.read_csv('ConductMice.csv')
print(data.columns)
data = data.drop(['Unnamed: 0'], axis=1)

Y = data[['target']]
X = data.drop(['target'], axis=1)

columns = list(X.columns)

test = pd.read_csv('test_data.csv')
Y_test = test[test.columns[-1]]
print(test.keys())
print(Y_test.shape)

X_test = test[columns]


Y_test = Y_test[:, np.newaxis]
X_test = np.nan_to_num(X_test)

print(Y_test.shape)
print(X_test.shape)

C = np.dot(X.T, X)
b = np.dot(X.T, Y)
theta = np.dot(np.linalg.inv(C + 0.1 * np.identity(X.shape[1])), b)

test_preds = np.dot(X_test, theta)
MSE_test = np.linalg.norm(test_preds - Y_test) ** 2 / X_test.shape[0]
print("Test MSE: ", MSE_test)
print("Test RMSE: ", sqrt(MSE_test))
print(sqrt(MSE_test) / sqrt(Y_test.var()))
