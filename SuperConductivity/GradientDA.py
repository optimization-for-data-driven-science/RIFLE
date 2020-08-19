import numpy as np
import pandas as pd
from numpy import genfromtxt
from math import sqrt
import matplotlib.pyplot as plt

confidence_matrix = genfromtxt('conf_matrix.csv', delimiter=',')
conf_list = genfromtxt('conf_list.csv', delimiter=',')


MSK_X = pd.read_csv('Conductor_Mask_X.csv')
MSK_Y = pd.read_csv('Conductor_Mask_Y.csv')

mskX = MSK_X.replace(0, np.nan)
mskY = MSK_Y.replace(0, np.nan)


MSK_X = MSK_X.values
MSK_Y = MSK_Y.values

TrainData = pd.read_csv('training_data.csv')

Y = TrainData[TrainData.columns[-1]]
Y = Y[:, np.newaxis]

X = TrainData.drop([TrainData.columns[-1]], axis=1)

Y_missing = np.multiply(MSK_Y, Y)
X_missing = np.multiply(MSK_X, X)

# Test Data
test = pd.read_csv('test_data.csv')
Y_test = test[test.columns[-1]]
Y_test = Y_test[:, np.newaxis]
X_test = test.drop([test.columns[-1]], axis=1)

C = np.dot(X_missing.T, X_missing) / np.dot(MSK_X.T, MSK_X)
b = np.dot(X_missing.T, Y_missing) / np.dot(MSK_X.T, MSK_Y)

const = 0.25
C_min = C - const * confidence_matrix
C_max = C + const * confidence_matrix

y_conf = np.asarray(conf_list)
y_conf = y_conf[:, np.newaxis]
print(y_conf.shape)
print(b.shape)
b_min = b - const * y_conf
b_max = b + const * y_conf

# step_size = 0.00000011
# number_of_iterations = 150000
lam = 0.0001

step_size = 1.1e-7
inner_step_size = 1.1e-9


number_of_iterations = 200000

vals = []
iterations = []

C1 = C
b1 = b
ident = np.identity(C.shape[0])

theta = np.dot(np.linalg.inv(C + lam * ident), b)

min_error = 10000
min_it = -1
# ones = np.ones(shape=(C.shape[0], C.shape[0]))
# ones_vector = np.ones(shape=(C.shape[0], 1))

for i in range(number_of_iterations):
    C1 += step_size * np.dot(theta, theta.T)
    """
    theta_gram = np.dot(theta, theta.T)
    sign_matrix = np.where(theta_gram > 0, 1, 0)
    sign2 = ones - sign_matrix
    C1 = np.multiply(C_max, sign_matrix) + np.multiply(C_min, sign2)

    sign_vector = np.where(theta > 0, 1, 0)
    sign_vector2 = ones_vector - sign_vector
    b1 = np.multiply(b_min, sign_vector) + np.multiply(b_max, sign_vector2)
    """

    # Applying box constraint:
    C1 = np.clip(C1, C_min, C_max)

    b1 += -2 * step_size * theta
    b1 = np.clip(b1, b_min, b_max)

    for _ in range(1):
        grad = np.dot(C1, theta) - b1 + lam * theta
        # print(grad[0][0])
        theta -= 2 * inner_step_size * grad

    # exit(0)
    val = np.dot(np.dot(theta.T, C1), theta) - np.dot(2*theta.T, b1) + lam * np.dot(theta.T, theta)
    if val > -100000:
        vals.append(val)
        iterations.append(i)

    if i % 10 == 9:
        print("------------------------")
        print("Iteration: ", i)
        test_preds = np.dot(X_test, theta)
        MSE_test = np.linalg.norm(test_preds - Y_test) ** 2 / X_test.shape[0]
        print("Test RMSE: ", sqrt(MSE_test))
        print(val)
        # V, S, U = np.linalg.svd(C1)

        # print(S[0])
        if sqrt(MSE_test) < min_error:
            min_error = sqrt(MSE_test)
            min_it = i

print("-----------*********-------")
print(min_it)
print(min_error)
print("-----------*********-------")
preds = np.dot(X, theta)
t = np.sum(MSK_Y)

preds = np.multiply(preds, MSK_Y)
MSE = np.linalg.norm(preds - Y_missing) ** 2 / t

print("Train MSE: ", MSE)
print("Train RMSE: ", sqrt(MSE))

test_preds = np.dot(X_test, theta)
MSE_test = np.linalg.norm(test_preds - Y_test) ** 2 / X_test.shape[0]
print("Test MSE: ", MSE_test)
print("Test RMSE: ", sqrt(MSE_test))
print("-----------------------------")

plt.scatter(iterations, vals)
plt.show()
