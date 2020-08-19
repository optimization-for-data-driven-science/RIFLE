from math import sqrt
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('feedback_data_missing80n10000.csv')


train_data = data.fillna(data.mean())

test_data = pd.read_csv('test_data.csv')

print(train_data)

print(test_data)
X = train_data.drop(['280'], axis=1)
Y = train_data[['280']]
X = X.to_numpy()
Y = Y.to_numpy()

lam = 8000
ident = np.identity(X.shape[1])

theta = np.dot(np.linalg.inv(np.dot(X.T, X) + lam * ident), np.dot(X.T, Y))

X_test = test_data.drop(['280'], axis=1)
Y_test = test_data[['280']]
X_test = X_test.to_numpy()
Y_test = Y_test.to_numpy()

print(X_test.shape)
print(Y_test.shape)

Y_hat = np.dot(X_test, theta)

mse = np.linalg.norm(Y_test - Y_hat)**2 / Y_test.shape[0]

print("Test MSE: ", mse)
print("Test RMSE: ", np.sqrt(mse))
