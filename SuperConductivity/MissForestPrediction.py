import pandas as pd
import numpy as np
import math
data = pd.read_csv('MissForestImputed.csv')
print(data.columns)
print(data.head())

Y = data[['167']]
X = data.drop(['167'], axis=1)

# X = X.to_numpy(X)
# Y = Y.to_numpy(Y)

print(X.shape)
print(Y.shape)
theta = np.dot(np.linalg.inv(np.dot(X.T, X) + 0.001 * np.identity(X.shape[1])), np.dot(X.T, Y))

preds = np.dot(X, theta)

MSE = np.linalg.norm(preds - Y) ** 2 / X.shape[0]

print("Train MSE: ", MSE)
print("Train RMSE: ", math.sqrt(MSE))

test = pd.read_csv('test_data.csv')
Y_test = test[test.columns[-1]]
print(test.keys())
print(Y_test.shape)

X_test = test.drop([test.columns[-1]], axis=1)
Y_test = Y_test[:, np.newaxis]


X_test = np.nan_to_num(X_test)
y_test_hat = np.dot(X_test, theta)

mse_test = np.linalg.norm(y_test_hat - Y_test)**2/Y_test.shape[0]
rmse_test = np.sqrt(mse_test)
print(mse_test)
print(rmse_test)

