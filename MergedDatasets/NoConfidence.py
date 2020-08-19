from math import sqrt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('missing_merged2_dataMNAR400.csv')

data_points = data.shape[0]

X = data[data.columns[:-1]]

mask_X = X.isna()

mask_X = mask_X.to_numpy()
mask_X = np.ones(shape=mask_X.shape) - mask_X

Y = data[['critical_temp']]

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

number_of_features = X.shape[1]
confidence_matrix = np.zeros(shape=(number_of_features, number_of_features))
features = train_X.columns

sample_coeff = 1
sampling_number = 30

with_replacement = True
# """
msk_gram = np.dot(mask_X.T, mask_X)
for i in range(number_of_features):
    print("Feature num: ", i + 1)
    for j in range(i, number_of_features):
        feature_i = features[i]
        feature_j = features[j]

        columns = train_X[[feature_i, feature_j]]
        intersections = columns[columns[[feature_i, feature_j]].notnull().all(axis=1)]

        intersection_num = len(intersections)
        if intersection_num != msk_gram[i][j]:
            print(intersection_num)
            print(msk_gram[i][j])
            print(i, j)
            print("Error")
            exit(1)

        sample_size = intersection_num // sample_coeff

        if sample_size < 2:
            max_vals = columns.max()
            max1 = max_vals[feature_i]
            max2 = max_vals[feature_j]
            confidence_matrix[i][j] = max1 * max2
            continue

        estimation_array = []

        for ind in range(sampling_number):
            current_sample = np.array(intersections.sample(n=sample_size, replace=with_replacement))
            with_replacement = False

            f1 = current_sample[:, 0]
            f2 = current_sample[:, 1]
            inner_prod = np.inner(f1, f2) / sample_size
            estimation_array.append(inner_prod)

        confidence_matrix[i][j] = np.std(estimation_array)
        # print(estimation_array)
        # print(i, j, C[i][j], confidence_matrix[i][j])

for j in range(number_of_features):
    for i in range(j + 1, number_of_features):
        confidence_matrix[i][j] = confidence_matrix[j][i]

print("------------Confidence Matrix---------------")
print(confidence_matrix)
print("---------------------------")
# target confidence:
conf_list = []

cov_msk_train = np.dot(mask_X.T, mask_Y)

for i in range(number_of_features):
    feature_i = features[i]
    current_feature = train_X[[feature_i]].to_numpy()
    current_Y = train_Y.to_numpy()

    columns = np.concatenate((current_feature, current_Y), axis=1)

    columns = pd.DataFrame(columns, columns=[feature_i, 'Y'])
    intersections = columns[columns[[feature_i, "Y"]].notnull().all(axis=1)]
    intersections2 = columns[columns[[feature_i]].notnull().all(axis=1)]

    intersection_num = len(intersections)
    intersection_num2 = len(intersections2)
    if intersection_num != cov_msk_train[i][0]:
        print(intersection_num, intersection_num2, cov_msk_train[i][0])
        exit(1)

    sample_size = intersection_num // sample_coeff
    estimation_array = []

    for ind in range(sampling_number):
        current_sample = np.array(intersections.sample(n=sample_size, replace=with_replacement))
        f1 = current_sample[:, 0]
        f2 = current_sample[:, 1]

        inner_prod = np.inner(f1, f2) / sample_size
        estimation_array.append(inner_prod)

    conf_list.append(np.std(estimation_array))

print(conf_list)

np.savetxt("conf_matrix.csv", confidence_matrix, delimiter=",")
np.savetxt("conf_list.csv", conf_list, delimiter=",")
# """

# confidence_matrix = np.loadtxt('conf_matrix.csv', delimiter=',')
# conf_list = np.loadtxt('conf_list.csv', delimiter=',')

print(confidence_matrix.shape)
print(conf_list)
const = 100
C_min = C - const * confidence_matrix
C_max = C + const * confidence_matrix

y_conf = np.asarray(conf_list)
y_conf = y_conf[:, np.newaxis]
print(y_conf.shape)
print(b.shape)

b_min = b - const * y_conf
b_max = b + const * y_conf

step_size = 0.01
number_of_iterations = 10000
lam = 0.5

theta = np.dot(np.linalg.inv(C + lam * np.identity(X.shape[1])), b)
ident = np.identity(C.shape[0])

check_every_iteration = False

print("----------------------")

predicts = []
for i in range(len(mask_X)):
    row_i = mask_X[i]
    nonzeros = np.nonzero(row_i)[0]
    nonzeros = list(nonzeros)

    data_X = X[:, nonzeros]

    currentMask = mask_X[:, nonzeros]
    res = currentMask.shape[1] * np.ones(currentMask.shape[0]) - np.sum(currentMask, axis=1)
    indices = np.argwhere(res < 3)
    indices = np.reshape(indices, (indices.shape[0], ))

    validation_indices = list(set(indices) & set(Y_not_nan))

    currentC = C[nonzeros, :]
    currentC = currentC[:, nonzeros]

    currentB = b[nonzeros, :]
    lam_list = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.8, 1, 1.5, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 50, 100]

    # Optimizing Current C and currentB
    X_validation = X[validation_indices, :]
    X_validation = X_validation[:, nonzeros]

    Y_validation = Y[validation_indices, :]

    best_lam = -1
    best_rmse = 1000000
    for lam in lam_list:
        theta = np.dot(np.linalg.inv(currentC + lam * np.identity(currentC.shape[0])), currentB)
        Y_predicted = np.dot(X_validation, theta)
        mse = np.linalg.norm(Y_validation - Y_predicted) ** 2 / Y_validation.shape[0]
        rmse = sqrt(mse)
        if rmse < best_rmse:
            best_rmse = rmse
            best_lam = lam

    theta = np.dot(np.linalg.inv(currentC + best_lam * np.identity(currentC.shape[0])), currentB)

    data_i = X[i]
    data_i = data_i[:, np.newaxis]
    data_i = data_i[nonzeros, :]

    y_predict = np.dot(data_i.T, theta)
    predicts.append(y_predict[0][0])
    print("Best Lambda: ", best_lam)
    print("Best RMSE:", best_rmse)
    print("-----------------------")

Y_pred = np.array(predicts)
Y_predictions = Y_pred[:, np.newaxis]
Y_predictions = Y_predictions * sqrt(sc_y.var_[0]) + sc_y.mean_[0]

original = pd.read_csv('original_merged2_dataMNAR400.csv')
Y_original = original[['critical_temp']]

original_std = sqrt(Y_original.var()[0])
Y_original = Y_original.to_numpy()

print(missing_entries)
mse = np.linalg.norm(np.multiply(Y_predictions - Y_original, mask_Y_test)) ** 2 / missing_entries
print("RMSE: ", sqrt(mse))
print("Scaled: ", sqrt(mse) / original_std)
