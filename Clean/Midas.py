import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from math import sqrt

train_data = pd.read_csv('MissForestImputed3000P95MCAR.csv')

print(train_data)
test_data = pd.read_csv('test_data.csv')

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
test_scale = sqrt(Y_test.var()[0])

lam_list = [0.01, 0.1, 1, 2, 5, 10, 20, 50, 100, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000]

inds = X_test.index
cols = X_test.columns

X_test = sc.transform(X_test)
Y_test = sc_y.transform(Y_test)

X_test_df = pd.DataFrame(X_test, index=inds, columns=cols)

for lam in lam_list:
    ident = np.identity(X.shape[1])

    theta = np.dot(np.linalg.inv(np.dot(X.T, X) + lam * ident), np.dot(X.T, Y))

    print(X_test.shape)
    print(Y_test.shape)

    Y_hat = np.dot(X_test, theta)

    mse = np.linalg.norm(Y_test - Y_hat) ** 2 / Y_test.shape[0]
    print("Lambda: ", lam)
    # print("MissForest:")
    # print("Test MSE: ", mse)
    # print("Test RMSE: ", np.sqrt(mse))
    print("Corrected Test RMSE: ", np.sqrt(mse) * ratio)
