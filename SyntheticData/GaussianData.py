import numpy as np

n_train = 500000
n = n_train + 10000
d = 50
k = 20

"""
number_of_non_zeros = 30
theta_mask = np.zeros((d, 1))
for i in range(number_of_non_zeros):
    theta_mask[i][0] = 1

np.random.shuffle(theta_mask)
print(theta_mask)
"""
S = np.random.normal(0, 1, k * d)
S = S.reshape((d, k))

Cov = np.dot(S, S.T)
mean = 200 * np.random.random(size=d) - 100

# print(Cov.shape)
# print(Cov)

X = np.random.multivariate_normal(mean, Cov, size=n)

# print(X)
# print(X.shape)

true_theta = np.random.random(size=(d, 1))

# true_theta = np.multiply(true_theta, theta_mask)

noise = np.random.normal(size=(n, 1), loc=0, scale=0.5)
noiseless_Y = np.dot(X, true_theta)

X_train = X[:n_train]
Y_train = noiseless_Y[:n_train]
X_test = X[n_train:]
Y_test = noiseless_Y[n_train:]

np.savetxt("X500K_d50.csv", X_train, delimiter=",")
np.savetxt("Y500K_d50.csv", Y_train, delimiter=",")

np.savetxt("X500K_d50_test.csv", X_test, delimiter=",")
np.savetxt("Y500K_d50_test.csv", Y_test, delimiter=",")

np.savetxt("Theta500K_d50.csv", true_theta, delimiter=",")
