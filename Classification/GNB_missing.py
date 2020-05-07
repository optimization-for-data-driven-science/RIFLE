import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from math import log
import math
import random


test = pd.read_csv('test_data.csv')

train_missing = pd.read_csv('train_missing80.csv')

Y_1 = train_missing.loc[train_missing['Y'] == 1]
Y_0 = train_missing.loc[train_missing['Y'] == 0]

y1_mean = Y_1.mean()
y0_mean = Y_0.mean()
print('-------------------')
y1_var = Y_1.var()
y0_var = Y_0.var()

Y = train_missing['Y']
X = train_missing.drop(['Y'], axis=1)

Y_test = test['Y']
X_test = test.drop(['Y'], axis=1)


y1_count = Y_1.shape[0]
y0_count = Y_0.shape[0]
print(y1_count)
print(y0_count)

c0 = log(y1_count / y0_count)

y1_var = y1_var.drop(['Y'])
y1_var = y1_var.to_numpy()

y0_var = y0_var.drop(['Y'])
y0_var = y0_var.to_numpy()

y1_mean = y1_mean.drop(['Y'])
y1_mean = y1_mean.to_numpy()

y0_mean = y0_mean.drop(['Y'])
y0_mean = y0_mean.to_numpy()

print(y1_var)
print(2*math.pi * y1_var)
print(np.reciprocal(2*math.pi * y1_var))
print(np.sqrt(np.reciprocal(2*math.pi * y1_var)))
print(np.log(np.sqrt(np.reciprocal(2*math.pi * y1_var))))
print(np.sum(np.log(np.sqrt(np.reciprocal(2*math.pi * y1_var)))))

c1 = np.sum(np.log(np.sqrt(np.reciprocal(2*math.pi * y1_var))))
c2 = np.sum(np.log(np.sqrt(np.reciprocal(2*math.pi * y0_var))))

print(c1)
print(c2)

constants = c0 + c1 - c2

testX = X_test.to_numpy()
preds = []
for k in range(Y_test.shape[0]):
    current_point = testX[k]
    """
    print(current_point)
    print(y1_mean)
    print(current_point - y1_mean)
    print(np.square(current_point - y1_mean))
    print(2*y1_var)
    print(np.square(current_point - y1_mean) / (2 * y1_var))
    print(np.sum(np.square(current_point - y1_mean) / (2 * y1_var)))
    """
    val1 = np.sum(np.square(current_point - y1_mean) / (2 * y1_var))
    val2 = np.sum(np.square(current_point - y0_mean) / (2 * y0_var))
    print(val1)
    print(val2)
    final_score = constants - val1 + val2
    print(final_score)
    if final_score > 0:
        preds.append(1)
    else:
        preds.append(0)

Y_hat = np.array(preds)
print(Y_hat.shape)
print(Y_test.shape)

Y_test = Y_test[:, np.newaxis]
Y_hat = Y_hat[:, np.newaxis]

print(np.sum(Y_test == Y_hat) / Y_test.shape[0])
