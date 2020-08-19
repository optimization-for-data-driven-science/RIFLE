from math import sqrt
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

MSK_X = pd.read_csv('Conductor_Mask_X.csv')
MSK_Y = pd.read_csv('Conductor_Mask_Y.csv')

mskX = MSK_X.replace(0, np.nan)
mskY = MSK_Y.replace(0, np.nan)

# print(mskY)
print(MSK_X.shape)
print(MSK_Y.shape)

MSK_X = MSK_X.values
MSK_Y = MSK_Y.values

TrainData = pd.read_csv('training_data.csv')

Y = TrainData[TrainData.columns[-1]]
# print(TrainData.keys())
print(Y.shape)

X = TrainData.drop([TrainData.columns[-1]], axis=1)
TrainNan = np.multiply(X, mskX)

TrainNan.to_csv('BlogDataMissing.csv', index=False)
# Mean Imputation
X_mean = TrainNan.fillna(TrainNan.mean())

Y = Y[:, np.newaxis]
YNan = np.multiply(Y, mskY)
Y_mean = YNan.fillna(YNan.mean())

# Test Data
test = pd.read_csv('test_data.csv')

print(test.head())
Y_test = test[test.columns[-1]]
print(test.shape)
exit(0)
X_test = test.drop([test.columns[-1]], axis=1)
print(X.shape)

Y_test = Y_test[:, np.newaxis]

print(sqrt(Y_test.var()))
exit(0)
Y_missing = np.multiply(MSK_Y, Y)
X_missing = np.multiply(MSK_X, X)

# Regular Linear Regression
C = np.dot(X.T, X)
b = np.dot(X.T, Y)

theta = np.dot(np.linalg.inv(C + 0.0001 * np.identity(X.shape[1])), b)

preds = np.dot(X, theta)

MSE = np.linalg.norm(preds - Y) ** 2 / X.shape[0]

print("Train MSE: ", MSE)
print("Train RMSE: ", sqrt(MSE))

test_preds = np.dot(X_test, theta)
MSE_test = np.linalg.norm(test_preds - Y_test) ** 2 / X_test.shape[0]
print("Test MSE: ", MSE_test)
print("Test RMSE: ", sqrt(MSE_test))

exit(0)
print("--------------------------------------")

t = np.sum(MSK_Y)

C = np.dot(X_missing.T, X_missing) / np.dot(MSK_X.T, MSK_X)
b = np.dot(X_missing.T, Y_missing) / np.dot(MSK_X.T, MSK_Y)

theta = np.dot(np.linalg.inv(C + 0.0001 * np.identity(X_missing.shape[1])), b)

preds = np.dot(X, theta)

preds = np.multiply(preds, MSK_Y)
MSE = np.linalg.norm(preds - Y_missing) ** 2 / t

print("Train MSE: ", MSE)
print("Train RMSE: ", sqrt(MSE))

test_preds = np.dot(X_test, theta)
MSE_test = np.linalg.norm(test_preds - Y_test) ** 2 / X_test.shape[0]
print("Test MSE: ", MSE_test)
print("Test RMSE: ", sqrt(MSE_test))

print("-------------------Imputation with mean-------------------\n")
theta = (
    np.dot(np.linalg.inv(np.dot(X_mean.T, X_mean) + 0.001 * np.identity(X_mean.shape[1])), np.dot(X_mean.T, Y_mean)))
preds = np.dot(X, theta)

preds = np.multiply(preds, MSK_Y)
MSE = np.linalg.norm(preds - Y_missing) ** 2 / t

print("Train MSE: ", MSE)
print("Train RMSE: ", sqrt(MSE))

test_preds = np.dot(X_test, theta)
MSE_test = np.linalg.norm(test_preds - Y_test) ** 2 / X_test.shape[0]
print("Test MSE: ", MSE_test)
print("Test RMSE: ", sqrt(MSE_test))

# print("----------------------------\n When just Y is available!\n")

# Confidence Intervals:

number_of_features = X.shape[1]
confidence_matrix = np.zeros(shape=(number_of_features, number_of_features))

features = TrainData.columns
print(features)

msk_gram = np.dot(MSK_X.T, MSK_X)

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

    sample_size = intersection_num // 20
    estimation_array = []

    for ind in range(20):
        current_sample = np.array(intersections.sample(n=sample_size))
        f1 = current_sample[:, 0]
        f2 = current_sample[:, 1]

        inner_prod = np.inner(f1, f2) / sample_size
        estimation_array.append(inner_prod)

    conf_list.append(np.std(estimation_array))

print(conf_list)

np.savetxt("conf_matrix20.csv", confidence_matrix, delimiter=",")
np.savetxt("conf_list20.csv", conf_list, delimiter=",")
