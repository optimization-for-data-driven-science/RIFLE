import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from math import sqrt

train_data = pd.read_csv('MissForestImputedNMAR60n300.csv')

print(train_data)
test_data = pd.read_csv('test.csv')

X = train_data[train_data.columns[0:-1]]
Y = train_data[[train_data.columns[-1]]]

Y_var = Y.var()[0]

print(X)
print(Y)

sc = StandardScaler()
sc_y = StandardScaler()

sc.fit(X)
sc_y.fit(Y)

X = sc.transform(X)
Y = sc_y.transform(Y)

var = sc_y.var_[0]
scale = sqrt(var)

# X = X.to_numpy()
# Y = Y.to_numpy()

print(X.shape)
print(Y.shape)

Y_test = test_data[[test_data.columns[-1]]]
X_test = test_data[test_data.columns[0:-1]]

ratio = sqrt(Y_var) / sqrt(Y_test.var()[0])

lam = 1000
ident = np.identity(X.shape[1])

theta = np.dot(np.linalg.inv(np.dot(X.T, X) + lam * ident), np.dot(X.T, Y))

test_scale = sqrt(Y_test.var()[0])
# X_test = sc.transform(X_test)
# Y_test = sc_y.transform(Y_test)

inds = X_test.index
cols = X_test.columns

X_test = sc.transform(X_test)
Y_test = sc_y.transform(Y_test)

X_test_df = pd.DataFrame(X_test, index=inds, columns=cols)

print(X_test.shape)
print(Y_test.shape)

Y_hat = np.dot(X_test, theta)

mse = np.linalg.norm(Y_test - Y_hat) ** 2 / Y_test.shape[0]

print("MissForest:")
print("Test MSE: ", mse)
print("Test RMSE: ", np.sqrt(mse))
print("Corrected Test RMSE: ", np.sqrt(mse) * ratio)
exit(0)
# -----------------------------------------------------------------------
train_data = pd.read_csv('SuperConductMICE60n300.csv')

"""
drops = [
         'entropy_atomic_radius',
         'wtd_range_Density',
         'wtd_gmean_Valence',
         'std_Valence']
"""
train_data = train_data.drop(['Unnamed: 0'], axis=1)

test_data = pd.read_csv('test_data.csv')
# test_data = test_data.drop(drops, axis=1)

Y_test = test_data[[test_data.columns[-1]]]
X_test = test_data[test_data.columns[0:-1]]

print(X_test)

X = train_data[train_data.columns[0:-1]]
Y = train_data[[train_data.columns[-1]]]

ratio = sqrt(Y.var()) / sqrt(Y_test.var())
print(X)

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
"""

lam = 4
ident = np.identity(X.shape[1])

theta = np.dot(np.linalg.inv(np.dot(X.T, X) + lam * ident), np.dot(X.T, Y))
Y_hat = np.dot(X_test, theta)

mse = np.linalg.norm(Y_test - Y_hat) ** 2 / Y_test.shape[0]

print("MICE:")
print("Test MSE: ", mse)
print("Test RMSE: ", np.sqrt(mse))
print("Corrected RMSE: ", np.sqrt(mse) * ratio)
exit(0)
