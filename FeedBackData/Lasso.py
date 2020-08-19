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

MSK_X = MSK_X.values
MSK_Y = MSK_Y.values

TrainData = pd.read_csv('blogData_train.csv', header=None)
train_index = TrainData.index
cols = TrainData.columns
TrainData = sc.fit_transform(TrainData)
TrainData = pd.DataFrame(TrainData, index=train_index, columns=cols)

Y = TrainData[TrainData.columns[-1]]

X = TrainData.drop([TrainData.columns[-1]], axis=1)
TrainNan = np.multiply(X, mskX)

Y = Y[:, np.newaxis]
YNan = np.multiply(Y, mskY)
YNan = YNan.fillna(-1)
TrainNan.to_csv('StandardBlogDataMissing.csv', index=False)
YNan.to_csv('StandardBlogDataMissingLabel.csv', index=False)
exit(0)
# Test Data
path = r'BlogFeedbackTest'
allFiles = glob.glob(path + "/*.csv")
test = pd.DataFrame()
for file_ in allFiles:
    test = test.append(pd.read_csv(file_, header=None))

test_index = test.index
cols = test.columns
test = sc.transform(test)
test = pd.DataFrame(test, index=test_index, columns=cols)

print(TrainData.head())
print(test.head())
TrainData.to_csv("StandardTrain.csv", index=False)
test.to_csv("StandardTest.csv", index=False)
exit(0)
Y_test = test[test.columns[-1]]

X_test = test.drop([test.columns[-1]], axis=1)

Y_test = Y_test[:, np.newaxis]

Y_missing = np.multiply(MSK_Y, Y)
X_missing = np.multiply(MSK_X, X)

# Regular Linear Regression
C = np.dot(X.T, X) / X.shape[0]
b = np.dot(X.T, Y) / X.shape[0]

theta = np.zeros(shape=(C.shape[0], 1))
number_of_iterations = 7000
lambda1 = 0.001
lambda2 = 0.001  # Set zero if you don't want to have L_2 regularizer
t_k = 0.001

shrinkage_parameter = lambda1 * t_k * np.ones(shape=theta.shape)
ones = np.ones(shape=theta.shape)

for i in range(number_of_iterations):
    grad = 2 * np.dot(C, theta) - 2 * b + 2 * lambda2 * theta
    shrinkage_input = theta - t_k * grad

    # Shrinkage
    temp = np.absolute(shrinkage_input) - shrinkage_parameter

    temp_sgn = (np.sign(temp) + ones) / 2
    val = np.multiply(temp, temp_sgn)
    theta = np.multiply(np.sign(shrinkage_input), val)

preds = np.dot(X, theta)

MSE = np.linalg.norm(preds - Y) ** 2 / X.shape[0]

print("Train MSE: ", MSE)
print("Train RMSE: ", sqrt(MSE))

test_preds = np.dot(X_test, theta)
MSE_test = np.linalg.norm(test_preds - Y_test) ** 2 / X_test.shape[0]
print("Test MSE: ", MSE_test)
print("Test RMSE: ", sqrt(MSE_test))
