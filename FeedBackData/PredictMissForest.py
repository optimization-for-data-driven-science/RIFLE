import pandas as pd
import numpy as np

data = pd.read_csv('ImputedWithMF.csv')
print(data.columns)

data = data.drop(['Unnamed: 0'], axis=1)

print(data.head(30))


Y = data[['280']]
X = data.drop(['280'], axis=1)

theta = np.dot(np.linalg.inv(np.dot(X.T, X) + 0.001 * np.identity(X.shape[1])), np.dot(X.T, Y))


standardTrain = pd.read_csv('StandardTrain.csv')
Y = standardTrain[standardTrain.columns[-1]]
Y = Y[:, np.newaxis]
X = standardTrain.drop([standardTrain.columns[-1]], axis=1)
X = np.nan_to_num(X)
print(Y.shape)
print(X.shape)

Y_hat = np.dot(X, theta)
mse_train = np.linalg.norm(Y_hat - Y)**2/Y.shape[0]
rmse_train = np.sqrt(mse_train)

print(mse_train)
print(rmse_train)

test = pd.read_csv('StandardTest.csv')
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

