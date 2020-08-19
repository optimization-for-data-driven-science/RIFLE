from math import sqrt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('missing_MNAR_P80_2000.csv')

data_points = data.shape[0]
column_numbers = []
i = 0

nulls = data.isnull().sum(axis=0)
for item in nulls:
    # print(item / data_points)
    if item / data_points > 0.95:
        column_numbers.append(data.columns[i])

    i += 1
print(column_numbers)
X = data[data.columns[0:-1]]

# print(X.head(5))
X_2 = X**2
X_2.columns = [str(col) + '2' for col in X_2.columns]


new_X1 = np.log(X + 1)
new_X1.columns = [str(col) + '_log' for col in new_X1.columns]

new_X2 = np.cos(X)
new_X2.columns = [str(col) + '_cos' for col in new_X2.columns]

new_X3 = np.sin(X)
new_X3.columns = [str(col) + '_sin' for col in new_X3.columns]

X = pd.concat([X, new_X1, new_X2, new_X3, X_2], axis=1)
# print(X)
# X = X.drop(column_numbers, axis=1)

mask_X = X.isna()

mask_X = mask_X.to_numpy()
mask_X = np.ones(shape=mask_X.shape) - mask_X

Y = data[['critical_temp']]

train_std = sqrt(Y.var()[0])


mask_Y_test = Y.isna()
mask_Y_test = mask_Y_test.to_numpy()
missing_entries = mask_Y_test.sum(axis=0)[0]
mask_Y = np.ones(shape=mask_Y_test.shape) - mask_Y_test
print(missing_entries)
print(mask_Y.shape)


sc = StandardScaler()
sc_y = StandardScaler()

sc.fit(X)
sc_y.fit(Y)


# print(sqrt(sc_y.var_))

cols = X.columns
inds = X.index

cols_y = Y.columns
inds_y = Y.index

X1 = sc.transform(X)
Y1 = sc_y.transform(Y)

X = sc.transform(X)
Y = sc_y.transform(Y)

train_X = pd.DataFrame(X1, columns=cols, index=inds)
train_Y = pd.DataFrame(Y1, columns=cols_y, index=inds_y)


# X = X.to_numpy()
# Y = Y.to_numpy()

X[np.isnan(X)] = 0
Y[np.isnan(Y)] = 0

C = np.dot(X.T, X) / np.dot(mask_X.T, mask_X)
b = np.dot(X.T, Y) / np.dot(mask_X.T, mask_Y)

# 1500000
lam = 6

predicts = []
for i in range(len(mask_X)):
    row_i = mask_X[i]
    nonzeros = np.nonzero(row_i)[0]
    nonzeros = list(nonzeros)

    currentC = C[nonzeros, :]
    currentC = currentC[:, nonzeros]

    currentB = b[nonzeros, :]
    theta = np.dot(np.linalg.inv(currentC + lam * np.identity(currentC.shape[0])), currentB)

    data_i = X[i]
    data_i = data_i[:, np.newaxis]
    data_i = data_i[nonzeros, :]

    y_predict = np.dot(data_i.T, theta)
    predicts.append(y_predict[0][0])

Y_pred = np.array(predicts)
Y_predictions = Y_pred[:, np.newaxis]
Y_predictions = Y_predictions * sqrt(sc_y.var_[0]) + sc_y.mean_[0]


original = pd.read_csv('original_MNAR_P80_2000.csv')
Y_original = original[['critical_temp']]

original_std = sqrt(Y_original.var()[0])
Y_original = Y_original.to_numpy()

mse = np.linalg.norm(np.multiply(Y_predictions - Y_original, mask_Y_test)) ** 2 / missing_entries
print("RMSE: ", sqrt(mse))
print("Scaled: ", sqrt(mse) / original_std)
exit(0)
print("------------------------------------")
theta = np.dot(np.linalg.inv(C + lam * np.identity(C.shape[0])), b)
imputed_data = pd.read_csv('MF_MNAR_P80_300.csv')
imputed_X = imputed_data[imputed_data.columns[0:-1]]
imputed_X = sc.transform(imputed_X)

predicted_y = np.dot(imputed_X, theta)
Y_predictions = predicted_y * sqrt(sc_y.var_[0]) + sc_y.mean_[0]
mse = np.linalg.norm(np.multiply(Y_predictions - Y_original, mask_Y_test)) ** 2 / missing_entries
print("RMSE: ", sqrt(mse))
print("Scaled: ", sqrt(mse) / original_std)
