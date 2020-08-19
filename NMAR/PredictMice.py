import pandas as pd
import numpy as np

train_data = pd.read_csv('NMARConductMice.csv')

print(train_data)

test_data = pd.read_csv('test_data.csv')

test_data = test_data.drop(['Po', 'At', 'Rn', 'H', 'F', 'Cl', 'Br', 'I', 'Nd', "He", "Ne", "Ar", "Kr", "Xe", "Pm"], axis=1)
print(test_data)
X = train_data.drop(['critical_temp', 'Unnamed: 0'], axis=1)
Y = train_data[['critical_temp']]
X = X.to_numpy()
Y = Y.to_numpy()

print(X.shape)
print(Y.shape)

lam = 400
ident = np.identity(X.shape[1])

theta = np.dot(np.linalg.inv(np.dot(X.T, X) + lam * ident), np.dot(X.T, Y))

X_test = test_data.drop(['critical_temp'], axis=1)
Y_test = test_data[['critical_temp']]
X_test = X_test.to_numpy()
Y_test = Y_test.to_numpy()

print(X_test.shape)
print(Y_test.shape)

Y_hat = np.dot(X_test, theta)

mse = np.linalg.norm(Y_test - Y_hat)**2 / Y_test.shape[0]

print("Test MSE: ", mse)
print("Test RMSE: ", np.sqrt(mse))
