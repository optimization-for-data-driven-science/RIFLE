import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

train_data = pd.read_csv('MissForestImputed80n500.csv')

print(train_data.shape)

test_data = pd.read_csv('test_data.csv')

# test_data = test_data.drop(['Po', 'At', 'Rn', 'H', 'F', 'Cl', 'Br', 'I', 'Nd'], axis=1)

X = train_data[train_data.columns[0:81]]
Y = train_data[[train_data.columns[-1]]]

print(X)
print(Y)
print(Y.shape)

sc = StandardScaler()
sc_y = StandardScaler()

sc.fit(X)
sc_y.fit(Y)


# X = sc.transform(X)
# Y = sc_y.transform(Y)

print(Y.shape)

X = X.to_numpy()
Y = Y.to_numpy()

print(X.shape)
print(Y.shape)

Y_test = test_data[[test_data.columns[-1]]]
X_test = test_data[test_data.columns[0:81]]

lam = 220000
ident = np.identity(X.shape[1])

theta = np.dot(np.linalg.inv(np.dot(X.T, X) + lam * ident), np.dot(X.T, Y))

# X_test = sc.transform(X_test)
# Y_test = sc_y.transform(Y_test)

X_test = X_test.to_numpy()
Y_test = Y_test.to_numpy()

print(X_test.shape)
print(Y_test.shape)

Y_hat = np.dot(X_test, theta)

mse = np.linalg.norm(Y_test - Y_hat) ** 2 / Y_test.shape[0]

print("MissForest:")
print("Test MSE: ", mse)
print("Test RMSE: ", np.sqrt(mse))
# -----------------------------------------------------------------------
train_data = pd.read_csv('MICE80n500.csv')

train_data = train_data.drop(['Unnamed: 0'], axis=1)
X = train_data[train_data.columns[0:81]]
Y = train_data[['critical_temp']]


Y_test = test_data[['critical_temp']]
X_test = test_data[test_data.columns[0:81]]

"""
sc = StandardScaler()
sc_y = StandardScaler()

sc.fit(X)
sc_y.fit(Y)


X = sc.transform(X)
Y = sc_y.transform(Y)

X_test = sc.transform(X_test)
Y_test = sc_y.transform(Y_test)
"""

X = X.to_numpy()
Y = Y.to_numpy()

X_test = X_test.to_numpy()
Y_test = Y_test.to_numpy()

lam = 4000
# lam = 10
ident = np.identity(X.shape[1])

theta = np.dot(np.linalg.inv(np.dot(X.T, X) + lam * ident), np.dot(X.T, Y))
Y_hat = np.dot(X_test, theta)

mse = np.linalg.norm(Y_test - Y_hat) ** 2 / Y_test.shape[0]

print("MICE:")
print("Test MSE: ", mse)
print("Test RMSE: ", np.sqrt(mse))

exit(0)
