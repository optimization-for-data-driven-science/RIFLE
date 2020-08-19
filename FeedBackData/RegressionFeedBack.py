from math import sqrt
import pandas as pd
import numpy as np
import glob
from sklearn.preprocessing import StandardScaler


sc = StandardScaler()


MSK_X = pd.read_csv('FeedBack_Mask_X.csv')
MSK_Y = pd.read_csv('FeedBack_Mask_Y.csv')


mskX = MSK_X.replace(0, np.nan)
mskY = MSK_Y.replace(0, np.nan)


# print(mskY)
print(MSK_X.shape)
print(MSK_Y.shape)

MSK_X = MSK_X.values
MSK_Y = MSK_Y.values

TrainData = pd.read_csv('blogData_train.csv', header=None)
# train_index = TrainData.index
# cols = TrainData.columns
# TrainData = sc.fit_transform(TrainData)
# TrainData = pd.DataFrame(TrainData, index=train_index, columns=cols)

Y = TrainData[TrainData.columns[-1]]
# print(TrainData.keys())
print(Y.shape)

X = TrainData.drop([TrainData.columns[-1]], axis=1)
TrainNan = np.multiply(X, mskX)

TrainNan.to_csv('BlogDataMissing.csv', index=False)
# Mean Imputation
X_mean = TrainNan.fillna(TrainNan.mean())
# print(TrainNan.head(30))
# print(X_mean.head(30))


Y = Y[:, np.newaxis]
print(Y.shape)
print(mskY.shape)
YNan = np.multiply(Y, mskY)
print(YNan.shape)
# YNan = YNan.fillna(-1)
# YNan.to_csv('BlogDataLabelMissing.csv', index=False)
# print(YNan.median)
# exit(0)
Y_mean = YNan.fillna(YNan.mean())

print(X.shape)
print(MSK_Y.shape)

# Test Data
path = r'BlogFeedbackTest'
allFiles = glob.glob(path + "/*.csv")
test = pd.DataFrame()
for file_ in allFiles:
    test = test.append(pd.read_csv(file_, header=None))

print(test)

test.to_csv('test_data.csv', index=False)
exit(0)

# test_index = test.index
# cols = test.columns
# test = sc.transform(test)
# test = pd.DataFrame(test, index=test_index, columns=cols)
print(test.head())
Y_test = test[test.columns[-1]]

X_test = test.drop([test.columns[-1]], axis=1)
print(X.shape)

Y_test = Y_test[:, np.newaxis]

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
theta = (np.dot(np.linalg.inv(np.dot(X_mean.T, X_mean) + 0.001 * np.identity(X_mean.shape[1])), np.dot(X_mean.T, Y_mean)))
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


const = 0.25
C_min = C - const * confidence_matrix
C_max = C + const * confidence_matrix

y_conf = np.asarray(conf_list)
y_conf = y_conf[:, np.newaxis]
print(y_conf.shape)
print(b.shape)
b_min = b - const * y_conf
b_max = b + const * y_conf

theta = np.zeros(shape=(C.shape[0], 1))

number_of_iterations = 600000
step_size = 0.00000003
lam = 0.001
ident = np.identity(C.shape[0])

min_error = 999999999999
min_it = -1

for i in range(number_of_iterations):
    C += step_size * np.dot(theta, theta.T)
    # Applying box constraint:
    C = np.clip(C, C_min, C_max)

    b += -2 * step_size * theta
    b = np.clip(b, b_min, b_max)

    theta = np.dot(np.linalg.inv(C + lam * ident), b)

    if i % 100 == 99:

        print("------------------------")
        print("Iteration: ", i)
        test_preds = np.dot(X_test, theta)
        MSE_test = np.linalg.norm(test_preds - Y_test) ** 2 / X_test.shape[0]
        print("Test RMSE: ", sqrt(MSE_test))
        if sqrt(MSE_test) < min_error:
            min_error = sqrt(MSE_test)
            min_it = i
        val = np.dot(np.dot(theta.T, C), theta) - np.dot(2*theta.T, b) + lam / 2 * np.dot(theta.T, theta)
        print(val)


print("-----------*********-------")
print(min_it)
print(min_error)
print("-----------*********-------")
np.savetxt("C_non_normalized.csv", C, delimiter=",")
np.savetxt("b_non_normalized.csv", b, delimiter=",")

theta2 = np.dot(np.linalg.inv(C + 0.000001 * ident), b)

# Evaluation
t = np.sum(MSK_Y)

preds = np.dot(X, theta)

preds = np.multiply(preds, MSK_Y)
MSE = np.linalg.norm(preds - Y_missing) ** 2 / t

print("Train MSE: ", MSE)
print("Train RMSE: ", sqrt(MSE))

test_preds = np.dot(X_test, theta)
MSE_test = np.linalg.norm(test_preds - Y_test) ** 2 / X_test.shape[0]
print("Test MSE: ", MSE_test)
print("Test RMSE: ", sqrt(MSE_test))

# --------------------

preds = np.dot(X, theta2)

preds = np.multiply(preds, MSK_Y)
MSE = np.linalg.norm(preds - Y_missing) ** 2 / t

print("Train MSE: ", MSE)
print("Train RMSE: ", sqrt(MSE))

test_preds = np.dot(X_test, theta2)
MSE_test = np.linalg.norm(test_preds - Y_test) ** 2 / X_test.shape[0]
print("Test MSE: ", MSE_test)
print("Test RMSE: ", sqrt(MSE_test))
