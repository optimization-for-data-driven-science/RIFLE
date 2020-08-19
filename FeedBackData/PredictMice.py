import pandas as pd
import numpy as np
import glob
from math import sqrt
from sklearn.preprocessing import StandardScaler

# sc = StandardScaler()


data = pd.read_csv('FeedBackMice.csv')
data = data.drop(['Unnamed: 0'], axis=1)

Y = data[['Y']]
X = data.drop(['Y'], axis=1)

cols_temp = list(X.columns)
cols = []
for item in cols_temp:
    name = item[1:]
    cols.append(int(name))

cols.append(280)
print(cols)

TrainData = pd.read_csv('blogData_train.csv', header=None)
TrainData = TrainData[cols]
Y_train = data[['Y']]
X_train = data.drop(['Y'], axis=1)

# train_index = TrainData.index
# columns = TrainData.columns
# TrainData = sc.fit_transform(TrainData)
# TrainData = pd.DataFrame(TrainData, index=train_index, columns=columns)


# train_index = data.index
# columns = data.columns
# data = sc.transform(data)
# data = pd.DataFrame(data, index=train_index, columns=columns)
# Y = data[['Y']]
# X = data.drop(['Y'], axis=1)


C = np.dot(X.T, X)
b = np.dot(X.T, Y)

theta = np.dot(np.linalg.inv(C + 0.1 * np.identity(X.shape[1])), b)

path = r'BlogFeedbackTest'
allFiles = glob.glob(path + "/*.csv")
test = pd.DataFrame()
for file_ in allFiles:
    test = test.append(pd.read_csv(file_, header=None))

test = test[cols]
# test_index = test.index
# columns = test.columns
# test = sc.transform(test)
# test = pd.DataFrame(test, index=test_index, columns=columns)

Y_test = test[test.columns[-1]]

X_test = test.drop([test.columns[-1]], axis=1)

Y_test = Y_test[:, np.newaxis]

print(Y_test.mean())
print(sqrt(Y_test.var()))
test_preds = np.dot(X_test, theta)
MSE_test = np.linalg.norm(test_preds - Y_test) ** 2 / X_test.shape[0]
print("Test MSE: ", MSE_test)
print("Test RMSE: ", sqrt(MSE_test))
print(sqrt(MSE_test) / sqrt(Y_test.var()))

t1 = 29.509097492137645/36.872057980432544
t2 = 84.04776302117261/36.872057980432544
t3 = 30.294935753952867/36.872057980432544
t4 = 31.25664147592128/36.872057980432544
t5 = 30.46407333315566/36.872057980432544
t6 = 31.09991536526746/36.872057980432544

print('Regular: ', t1, t1*t1)
print('Point Estimation: ', t2, t2*t2)
print('Confidence Intervals: ', t3, t3*t3)
print('Mean: ', t4, t4*t4)
print('MissForest: ', t5, t5*t5)
print('Median: ', t6, t6*t6)
