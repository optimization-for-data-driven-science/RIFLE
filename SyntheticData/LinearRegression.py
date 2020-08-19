from math import sqrt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('data_mcar60_d50_200K.csv')
d = str(50)


X_test = np.loadtxt('X200K_d50_test.csv', delimiter=',')
Y_test = np.loadtxt('Y200K_d50_test.csv', delimiter=',')
Y_test = Y_test[:, np.newaxis]

number_of_test_points = Y_test.shape[0]
original_std = np.nanstd(Y_test)


data_points = data.shape[0]
column_numbers = []
i = 0

nulls = data.isnull().sum(axis=0)
for item in nulls:
    # print(item / data_points)
    if item / data_points > 0.95:
        column_numbers.append(data.columns[i])

    i += 1
X = data[data.columns[0:-1]]

# X = X.drop(column_numbers, axis=1)

mask_X = X.isna()

mask_X = mask_X.to_numpy()
mask_X = np.ones(shape=mask_X.shape) - mask_X

Y = data[[d]]

train_std = sqrt(Y.var()[0])
train_mean = Y.mean()[0]

mask_Y_test = Y.isna()
mask_Y_test = mask_Y_test.to_numpy()
missing_entries = mask_Y_test.sum(axis=0)[0]
mask_Y = np.ones(shape=mask_Y_test.shape) - mask_Y_test

sc = StandardScaler()
sc_y = StandardScaler()

sc.fit(X)
sc_y.fit(Y)

cols = X.columns
inds = X.index

cols_y = Y.columns
inds_y = Y.index

X1 = sc.transform(X)
Y1 = sc_y.transform(Y)

X = sc.transform(X)
Y = sc_y.transform(Y)

X_test = sc.transform(X_test)

train_X = pd.DataFrame(X1, columns=cols, index=inds)
train_Y = pd.DataFrame(Y1, columns=cols_y, index=inds_y)

X[np.isnan(X)] = 0
Y[np.isnan(Y)] = 0

C = np.dot(X.T, X) / np.dot(mask_X.T, mask_X)
b = np.dot(X.T, Y) / np.dot(mask_X.T, mask_Y)

number_of_features = X.shape[1]
confidence_matrix = np.zeros(shape=(number_of_features, number_of_features))
features = train_X.columns
print(features)

sample_coeff = 1
sampling_number = 30

step_size = 0.00001
number_of_iterations = 5000

lam_list = [0.00001, 0.00002, 0.00005, 0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.12, 0.15, 0.18, 0.2, 0.5, 1, 2, 5, 10]

print("----------------------")
for lam in lam_list:

    theta = np.dot(np.linalg.inv(C + lam * np.identity(X.shape[1])), b)

    Y_pred = np.dot(X_test, theta)
    Y_pred = train_std * Y_pred + train_mean
    mse = np.linalg.norm(Y_pred - Y_test) ** 2 / number_of_test_points
    print(lam)
    print("RMSE: ", sqrt(mse))
    print("Scaled: ", sqrt(mse) / original_std)
    print("-----------------------------")

