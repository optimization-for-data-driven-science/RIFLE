import numpy as np
import random


n = 2000
d = 200

alpha_vector = []
for i in range(d):
    alpha_vector.append(1 * random.random() + 0)
# X = np.random.multivariate_normal(mean, Cov, size=n)

X = np.random.dirichlet(alpha_vector, size=n)

true_theta = 10 * np.random.random(size=(d, 1))

noise = np.random.normal(size=(n, 1), loc=0, scale=0.001)
noiseless_Y = np.dot(X, true_theta)
print(noiseless_Y)
true_Y = np.dot(X, true_theta) + noise

# print(noiseless_Y)
# print(true_Y)
# exit(0)
print(X.shape)
print(true_Y.shape)
# np.savetxt("Beta_X2000.csv", X, delimiter=",")
# np.savetxt("Beta_Y2000.csv", true_Y, delimiter=",")
# np.savetxt("Theta2000.csv", true_theta, delimiter=",")
