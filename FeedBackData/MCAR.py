import numpy as np
import pandas as pd
import random
import glob
from math import sqrt

missingValueRate = 0.3
trainData = pd.read_csv('blogData_train.csv', header=None)

n, d = trainData.shape

mask = np.ones((n, d))

for i in range(n):
    for j in range(d-1):
        t = random.random()
        if t < missingValueRate:
            mask[i][j] = 0

Y = trainData[trainData.columns[-1]]
Y = Y[:, np.newaxis]
X = trainData.drop([trainData.columns[-1]], axis=1)

msk_X = mask[:, :-1]
msk_Y = mask[:, -1]
msk_Y = msk_Y[:, np.newaxis]

print(msk_X.shape)
print(msk_Y.shape)

mskX = msk_X
mskX[mskX == 0] = np.nan
mskY = msk_Y
mskY[mskY == 0] = np.nan


# Test Data
path = r'BlogFeedbackTest'
allFiles = glob.glob(path + "/*.csv")
test = pd.DataFrame()
for file_ in allFiles:
    test = test.append(pd.read_csv(file_, header=None))

Y_test = test[test.columns[-1]]
Y_test = Y_test[:, np.newaxis]
X_test = test.drop([test.columns[-1]], axis=1)


Y_missing = np.multiply(msk_Y, Y)
X_missing = np.multiply(msk_X, X)

TrainNan = np.multiply(X, mskX)
YNan = np.multiply(Y, mskY)

C = np.dot(X_missing.T, X_missing) / np.dot(msk_X.T, msk_X)
b = np.dot(X_missing.T, Y_missing) / np.dot(msk_X.T, msk_Y)

theta = np.dot(np.linalg.inv(C + 0.0001 * np.identity(X_missing.shape[1])), b)

preds = np.dot(X, theta)

preds = np.multiply(preds, msk_Y)
t = np.sum(msk_Y)
MSE = np.linalg.norm(preds - Y_missing) ** 2 / t

print("Train MSE: ", MSE)
print("Train RMSE: ", sqrt(MSE))

test_preds = np.dot(X_test, theta)
MSE_test = np.linalg.norm(test_preds - Y_test) ** 2 / X_test.shape[0]
print("Test MSE: ", MSE_test)
print("Test RMSE: ", sqrt(MSE_test))

# Confidence Intervals:

number_of_features = X.shape[1]
confidence_matrix = np.zeros(shape=(number_of_features, number_of_features))

features = list(X.columns)
print(features)
exit(0)

msk_gram = np.dot(msk_X.T, msk_X)
for i in range(number_of_features):
    print("Feature num: ", i + 1)
    for j in range(i, number_of_features):
        feature_i = features[i]
        feature_j = features[j]

        columns = TrainNan[[feature_i, feature_j]]

        intersections = columns[columns[[feature_i, feature_j]].notnull().all(axis=1)]

        intersection_num = len(intersections)
        if intersection_num != msk_gram[i][j]:
            print(intersection_num)
            print(msk_gram[i][j])
            print(i, j)
            print("Error")
            exit(1)

        sample_size = intersection_num // 10
        estimation_array = []

        for ind in range(10):
            current_sample = np.array(intersections.sample(n=sample_size))

            f1 = current_sample[:, 0]
            f2 = current_sample[:, 1]

            inner_prod = np.inner(f1, f2) / sample_size
            estimation_array.append(inner_prod)

        confidence_matrix[i][j] = np.std(estimation_array)

for j in range(number_of_features):
    for i in range(j + 1, number_of_features):
        confidence_matrix[i][j] = confidence_matrix[j][i]

# target confidence:
conf_list = []

cov_msk_train = np.dot(MSK_X.T, MSK_Y)

for i in range(number_of_features):
    feature_i = features[i]
    current_feature = TrainNan[[feature_i]].to_numpy()
    current_Y = YNan.to_numpy()

    columns = np.concatenate((current_feature, current_Y), axis=1)

    columns = pd.DataFrame(columns, columns=[feature_i, 'Y'])
    intersections = columns[columns[[feature_i, "Y"]].notnull().all(axis=1)]
    intersections2 = columns[columns[[feature_i]].notnull().all(axis=1)]

    intersection_num = len(intersections)
    intersection_num2 = len(intersections2)
    if intersection_num != cov_msk_train[i][0]:
        print(intersection_num, intersection_num2, cov_msk_train[i][0])
        exit(1)

    sample_size = intersection_num // 10
    estimation_array = []

    for ind in range(10):
        current_sample = np.array(intersections.sample(n=sample_size))
        f1 = current_sample[:, 0]
        f2 = current_sample[:, 1]

        inner_prod = np.inner(f1, f2) / sample_size
        estimation_array.append(inner_prod)

    conf_list.append(np.std(estimation_array))

print(conf_list)

