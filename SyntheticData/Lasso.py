from math import sqrt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

column_name = '200'
data = pd.read_csv('sparse_data_mcar40.csv')
print(data)
column_names = list(data.columns)
print(column_names)
rmses = []

data_points = data.shape[0]
dimension = data.shape[1]
validation_threshold = dimension // 10
# print(validation_threshold)
X = data.drop([column_name], axis=1)

mask_X = X.isna()

mask_X = mask_X.to_numpy()
mask_X = np.ones(shape=mask_X.shape) - mask_X

Y = data[[column_name]]

train_std = sqrt(Y.var()[0])

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

Y_not_nan = np.nonzero(np.ones(Y.shape) - np.isnan(Y))[0]

np.isnan(Y)

train_X = pd.DataFrame(X1, columns=cols, index=inds)
train_Y = pd.DataFrame(Y1, columns=cols_y, index=inds_y)

X[np.isnan(X)] = 0
Y[np.isnan(Y)] = 0

mask_gram = np.dot(mask_X.T, mask_X)
mask_gram = np.where(mask_gram == 0, 1, mask_gram)

C = np.dot(X.T, X) / mask_gram
b = np.dot(X.T, Y) / np.dot(mask_X.T, mask_Y)

Y_original = np.loadtxt('Sparse_Y2000.csv', delimiter=',')
Y_original = Y_original[:, np.newaxis]

predicts = []
for i in range(len(mask_X)):

    # print(data[['critical_temp']].loc[i][0])
    if not np.isnan(data[[column_name]].loc[i][0]):
        predicts.append(data[[column_name]].loc[i][0])
        # print("Continue")
        # print("-------------------------")
        continue

    row_i = mask_X[i]
    nonzeros = np.nonzero(row_i)[0]
    nonzeros = list(nonzeros)

    data_X = X[:, nonzeros]

    currentMask = mask_X[:, nonzeros]
    res = currentMask.shape[1] * np.ones(currentMask.shape[0]) - np.sum(currentMask, axis=1)

    data_i = X[i]
    data_i = data_i[:, np.newaxis]
    data_i = data_i[nonzeros, :]

    currentC = C[nonzeros, :]
    currentC = currentC[:, nonzeros]

    currentB = b[nonzeros, :]

    theta = np.zeros(shape=(currentC.shape[0], 1))
    number_of_iterations = 7000
    lambda1 = 0.07
    lambda2 = 0.0  # Set zero if you don't want to have L_2 regulari
    t_k = 0.001

    shrinkage_parameter = lambda1 * t_k * np.ones(shape=theta.shape)
    ones = np.ones(shape=theta.shape)

    for _ in range(number_of_iterations):
        grad = 2 * np.dot(currentC, theta) - 2 * currentB + 2 * lambda2 * theta
        shrinkage_input = theta - t_k * grad

        # Shrinkage
        temp = np.absolute(shrinkage_input) - shrinkage_parameter

        temp_sgn = (np.sign(temp) + ones) / 2
        val = np.multiply(temp, temp_sgn)
        theta = np.multiply(np.sign(shrinkage_input), val)

    # print(theta)
    y_predict = np.dot(data_i.T, theta)
    y_with_confidence = y_predict[0][0] * sqrt(sc_y.var_[0]) + sc_y.mean_[0]

    # print("With confidence loss: ", abs(y_with_confidence - actual_y))

    predicts.append(y_predict[0][0])

Y_pred = np.array(predicts)
Y_predictions = Y_pred[:, np.newaxis]
Y_predictions = Y_predictions * sqrt(sc_y.var_[0]) + sc_y.mean_[0]


original_std = sqrt(Y_original.var())
mse = np.linalg.norm(np.multiply(Y_predictions - Y_original, mask_Y_test)) ** 2 / missing_entries
print("RMSE: ", sqrt(mse))
print("Scaled: ", sqrt(mse) / original_std)
rmses.append(sqrt(mse) / original_std)
print("---------------------------------------")
