import numpy as np
import pandas as pd
from numpy import genfromtxt
import glob

C = genfromtxt('C_non_normalized.csv', delimiter=',')
b = genfromtxt('b_non_normalized.csv', delimiter=',')


# Test Data
path = r'BlogFeedbackTest'
allFiles = glob.glob(path + "/*.csv")
test = pd.DataFrame()
for file_ in allFiles:
    test = test.append(pd.read_csv(file_, header=None))


# test_index = test.index
# cols = test.columns
# test = sc.transform(test)
# test = pd.DataFrame(test, index=test_index, columns=cols)
print(test.head())
Y_test = test[test.columns[-1]]

X_test = test.drop([test.columns[-1]], axis=1)

test_preds = np.dot(X_test, theta)
MSE_test = np.linalg.norm(test_preds - Y_test) ** 2 / X_test.shape[0]
print("Test MSE: ", MSE_test)
print("Test RMSE: ", sqrt(MSE_test))

print(C)
