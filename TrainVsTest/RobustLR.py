from math import sqrt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


number_of_iterations = 2000

predicts1 = []
predicts2 = []

avg1 = 0
avg2 = 0

rmse_avg1 = 0
rmse_avg2 = 0

data = pd.read_csv('missing_Beta_MCAR_85_10_300.csv')
test_data = pd.read_csv("test_Beta_MCAR_85_2000_300.csv")

data_points = data.shape[0]
dimension = data.shape[1]
validation_threshold = dimension // 5

print(validation_threshold)
X = data[data.columns[:-1]]

test_X = test_data[data.columns[:-1]]

number_of_test_points = test_X.shape[0]

nulls = X.isnull().sum()

null_cols = []
columns2 = []
for i in range(len(nulls)):
    if nulls[i] >= X.shape[0]-1:
        columns2.append(X.columns[i])

X = X.drop(columns2, axis=1)
test_X = test_X.drop(columns2, axis=1)

mask_X = X.isna()
mask_X = mask_X.to_numpy()
mask_X = np.ones(shape=mask_X.shape) - mask_X

mask_X_test = test_X.isna()
mask_X_test = mask_X_test.to_numpy()
mask_X_test = np.ones(shape=mask_X_test.shape) - mask_X_test

Y = data[['critical_temp']]
test_Y = test_data[['critical_temp']]

train_std = sqrt(Y.var()[0])
test_std = sqrt(test_Y.var()[0])

mask_Y_test = Y.isna()
mask_Y_test = mask_Y_test.to_numpy()
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

test_X = sc.transform(test_X)
# test_Y = sc.transform(test_Y)

Y_not_nan = np.nonzero(np.ones(Y.shape) - np.isnan(Y))[0]

train_X = pd.DataFrame(X1, columns=cols, index=inds)
train_Y = pd.DataFrame(Y1, columns=cols_y, index=inds_y)

X[np.isnan(X)] = 0
Y[np.isnan(Y)] = 0

mask_gram = np.dot(mask_X.T, mask_X)
mask_gram = np.where(mask_gram == 0, 1, mask_gram)

cov_gram = np.dot(mask_X.T, mask_Y)
cov_gram = np.where(cov_gram == 0, 1, cov_gram)

C = np.dot(X.T, X) / mask_gram
b = np.dot(X.T, Y) / cov_gram

number_of_features = X.shape[1]
confidence_matrix = np.zeros(shape=(number_of_features, number_of_features))
features = train_X.columns

sample_coeff = 1
sampling_number = 30

C_min = np.zeros(C.shape)
C_max = np.zeros(C.shape)

b_max = np.zeros(b.shape)
b_min = np.zeros(b.shape)

with_replacement = True
msk_gram = np.dot(mask_X.T, mask_X)
for i in range(number_of_features):
    print("Feature num: ", i + 1)
    for j in range(i, number_of_features):
        feature_i = features[i]
        feature_j = features[j]

        columns = train_X[[feature_i, feature_j]]
        # if i == 26:
        #     print(feature_i)
        #     print(columns)
        #     exit(0)
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
            C_max[i][j] = max1 * max2
            C_min[i][j] = - max1 * max2
            continue

        estimation_array = []

        for ind in range(sampling_number):
            current_sample = np.array(intersections.sample(n=sample_size, replace=with_replacement))
            with_replacement = False

            f1 = current_sample[:, 0]
            f2 = current_sample[:, 1]
            inner_prod = np.inner(f1, f2) / sample_size
            estimation_array.append(inner_prod)

        # confidence_matrix[i][j] = np.std(estimation_array)

        C_min[i][j] = min(estimation_array)
        C_max[i][j] = max(estimation_array)

        # print(estimation_array)
        # print(i, j, C[i][j], confidence_matrix[i][j])

for j in range(number_of_features):
    for i in range(j + 1, number_of_features):
        # confidence_matrix[i][j] = confidence_matrix[j][i]
        C_min[i][j] = C_min[j][i]
        C_max[i][j] = C_max[j][i]

# print("------------Confidence Matrix---------------")
# print(confidence_matrix)
# print("---------------------------")
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

    if sample_size < 2:
        max_vals = columns.max()
        max1 = max_vals[feature_i]
        max2 = max_vals["Y"]

        b_max[i][0] = max1 * max2
        b_min[i][0] = - max1 * max2
        # conf_list.append(max1 * max2)
        continue

    estimation_array = []

    for ind in range(sampling_number):
        current_sample = np.array(intersections.sample(n=sample_size, replace=with_replacement))
        f1 = current_sample[:, 0]
        f2 = current_sample[:, 1]

        inner_prod = np.inner(f1, f2) / sample_size
        estimation_array.append(inner_prod)

    # conf_list.append(np.std(estimation_array))
    b_max[i][0] = max(estimation_array)
    b_min[i][0] = min(estimation_array)

# print(conf_list)

# np.savetxt("conf_matrix.csv", confidence_matrix, delimiter=",")
# np.savetxt("conf_list.csv", conf_list, delimiter=",")

# confidence_matrix = np.loadtxt('conf_matrix.csv', delimiter=',')
# conf_list = np.loadtxt('conf_list.csv', delimiter=',')

# print(confidence_matrix.shape)
# print(conf_list)
# const = 1
# C_min = C - const * confidence_matrix
# C_max = C + const * confidence_matrix

# y_conf = np.asarray(conf_list)
# y_conf = y_conf[:, np.newaxis]
# print(y_conf.shape)
print(b.shape)

# b_min = b - const * y_conf
# b_max = b + const * y_conf

print("----------------------")

predicts = []


for i in range(number_of_test_points):

    row_i = mask_X_test[i]
    nonzeros = np.nonzero(row_i)[0]
    nonzeros = list(nonzeros)

    currentMask = mask_X[:, nonzeros]
    res = currentMask.shape[1] * np.ones(currentMask.shape[0]) - np.sum(currentMask, axis=1)

    size = 0
    counter = 0
    while size < validation_threshold:
        indices = np.argwhere(res < validation_threshold + counter)
        indices = np.reshape(indices, (indices.shape[0], ))
        size = len(indices)
        counter += 1

    validation_indices = list(set(indices) & set(Y_not_nan))

    currentC = C[nonzeros, :]
    currentC = currentC[:, nonzeros]

    currentB = b[nonzeros, :]
    lam_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.8, 1, 1.5, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 50, 100]

    # Optimizing Current C and currentB
    X_validation = X[validation_indices, :]
    X_validation = X_validation[:, nonzeros]
    X_validation = np.nan_to_num(X_validation)
    Y_validation = Y[validation_indices, :]

    best_lam = 1
    best_rmse = 1000000
    for lam in lam_list:
        theta = np.dot(np.linalg.inv(currentC + lam * np.identity(currentC.shape[0])), currentB)
        Y_predicted = np.dot(X_validation, theta)
        mse = np.linalg.norm(Y_validation - Y_predicted) ** 2 / Y_validation.shape[0]
        rmse = sqrt(mse)
        if rmse < best_rmse:
            best_rmse = rmse
            best_lam = lam

    print("Without Best res:", best_rmse)
    theta = np.dot(np.linalg.inv(currentC + best_lam * np.identity(currentC.shape[0])), currentB)

    data_i = test_X[i]
    data_i = data_i[:, np.newaxis]
    data_i = data_i[nonzeros, :]

    y_predict = np.dot(data_i.T, theta)
    y_without_confidence = y_predict[0][0] * sqrt(sc_y.var_[0]) + sc_y.mean_[0]
    actual_y = test_Y.loc[i][0]

    without = y_predict[0][0]
    best_solution1 = abs(y_without_confidence - actual_y)
    print("Without confidence loss: ", abs(y_without_confidence - actual_y))
    avg1 += best_solution1
    rmse_avg1 += best_solution1 * best_solution1

    # Now Solving min-max
    step_size = best_lam / 100

    currentCmin = C_min[nonzeros, :]
    currentCmin = currentCmin[:, nonzeros]

    currentCmax = C_max[nonzeros, :]
    currentCmax = currentCmax[:, nonzeros]

    currentBmin = b_min[nonzeros, :]
    currentBmax = b_min[nonzeros, :]
    ident = np.identity(currentCmax.shape[0])

    best_res = 9999
    bestB = currentB
    bestC = currentC

    for k in range(number_of_iterations):

        currentC += step_size * np.dot(theta, theta.T)
        currentC = np.clip(currentC, currentCmin, currentCmax)

        currentB += -2 * step_size * theta
        currentB = np.clip(currentB, currentBmin, currentBmax)

        theta = np.dot(np.linalg.inv(currentC + best_lam * ident), currentB)

        if k % 100 == 1:
            new_predictions = np.dot(X_validation, theta)
            mse = np.linalg.norm(new_predictions - Y_validation) ** 2 / Y_validation.shape[0]
            if sqrt(mse) < best_res:
                best_res = sqrt(mse)
                bestB = currentB
                bestC = currentC
        # val = np.dot(np.dot(theta.T, currentC), theta) - np.dot(2 * theta.T, currentB) + best_lam * np.dot(theta.T, theta)

    print("With Best res:", best_res)
    theta = np.dot(np.linalg.inv(bestC + best_lam * np.identity(currentC.shape[0])), bestB)
    # theta = np.dot(np.linalg.inv(currentC + best_lam * np.identity(currentC.shape[0])), currentB)

    y_predict = np.dot(data_i.T, theta)
    y_with_confidence = y_predict[0][0] * sqrt(sc_y.var_[0]) + sc_y.mean_[0]
    with_c = y_predict[0][0]

    best_solution2 = abs(y_with_confidence - actual_y)
    print("With confidence loss: ", best_solution2)
    avg2 += best_solution2
    rmse_avg2 += best_solution2 * best_solution2

    if 1.2 * best_rmse < best_res:
        predicts1.append(without)
    else:
        predicts1.append(with_c)

    predicts2.append(without)

    predicts.append(with_c)
    print("Best Lambda: ", best_lam)
    print("Best RMSE:", best_rmse)
    print("Data: ", i)
    print("-----------------------")

Y_pred = np.array(predicts)
Y_predictions = Y_pred[:, np.newaxis]
Y_predictions = Y_predictions * sqrt(sc_y.var_[0]) + sc_y.mean_[0]

original_std = sqrt(test_Y.var()[0])
test_Y = test_Y.to_numpy()

mse = np.linalg.norm(Y_predictions - test_Y) ** 2 / test_Y.shape[0]
print("RMSE: ", sqrt(mse))
print("Scaled: ", sqrt(mse) / original_std)
print("---------------------------")

print("With Confidence:")
Y_pred = np.array(predicts1)
Y_predictions = Y_pred[:, np.newaxis]
Y_predictions = Y_predictions * sqrt(sc_y.var_[0]) + sc_y.mean_[0]

mse = np.linalg.norm(Y_predictions - test_Y) ** 2 / test_Y.shape[0]
print("RMSE: ", sqrt(mse))
print("Scaled: ", sqrt(mse) / original_std)
print("---------------------------")

print("Without Confidence:")
Y_pred = np.array(predicts2)
Y_predictions = Y_pred[:, np.newaxis]
Y_predictions = Y_predictions * sqrt(sc_y.var_[0]) + sc_y.mean_[0]

mse = np.linalg.norm(Y_predictions - test_Y) ** 2 / test_Y.shape[0]
print("RMSE: ", sqrt(mse))
print("Scaled: ", sqrt(mse) / original_std)
print("-----------------------------")
print(avg1)
print(avg2)
print(original_std)
print("--------------------")
print(avg1 / 2000)
print(avg2 / 2000)
print("--------------------")
print(avg1 / 2000 / original_std)
print(avg2 / 2000 / original_std)
print("------------------------------------")
print(sqrt(rmse_avg1 / 2000) / original_std)
print(sqrt(rmse_avg2 / 2000) / original_std)


print("Number of iterations: ", number_of_iterations)
