from math import sqrt
import pandas as pd
import numpy as np

TrainData = pd.read_csv('training_data.csv')

Y = TrainData[TrainData.columns[-1]]
# print(TrainData.keys())
print(Y.shape)

X = TrainData.drop([TrainData.columns[-1]], axis=1)
Y = Y[:, np.newaxis]

# Test Data
test = pd.read_csv('test_data.csv')

Y_test = test[test.columns[-1]]

X_test = test.drop([test.columns[-1]], axis=1)

Y_test = Y_test[:, np.newaxis]

# Regular Linear Regression
C = np.dot(X.T, X) / 1000000000000
b = np.dot(X.T, Y) / 1000000000000

theta = np.zeros(shape=(C.shape[0], 1))
number_of_iterations = 8000
lambda1 = 0.000008
lambda2 = 0  # Set zero if you don't want to have L_2 regularizer
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
