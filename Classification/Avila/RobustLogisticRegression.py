import numpy as np
import pandas as pd
from math import fabs
from sklearn.preprocessing import StandardScaler

number_of_samples = 1000


def sigmoid(inp):
    return 1.0 / (1.0 + np.exp(-inp))


def find_mean_and_covariance(data_set, sample_num=-1):
    if sample_num == -1:
        sample_num = data_set.shape[0]

    cov_matrix = np.zeros(shape=(data_set.shape[1], data_set.shape[1]))
    mu = np.zeros(shape=(data_set.shape[1],))

    for i in range(data_set.shape[1]):
        for j in range(i + 1):
            feature_i = data_set.columns[i]
            feature_j = data_set.columns[j]

            columns = data_set[[feature_i, feature_j]]

            intersections = columns[columns[[feature_i, feature_j]].notnull().all(axis=1)]

            if intersections.shape[0] < 2:
                pass
            sample = intersections.sample(sample_num, replace=True).to_numpy()
            f1 = sample[:, 0]
            f2 = sample[:, 1]

            inner_prod = np.inner(f1, f2) / sample_num
            f1_mean = f1.mean()
            f2_mean = f2.mean()
            cov_estimation = inner_prod - f1_mean * f2_mean
            cov_matrix[i][j] = cov_estimation

    for i in range(data_set.shape[1]):
        for j in range(i):
            cov_matrix[i][j] = cov_matrix[j][i]

    for i in range(data_set.shape[1]):
        feature_i = data_set.columns[i]
        column = data_set[[feature_i]]
        mean = column.mean()
        mu[i] = list(mean)[0]

    return mu, cov_matrix


train = pd.read_csv('train_missing_MCAR40_100_noy_1.csv')
test = pd.read_csv('avila-ts.csv')

test.loc[test['11'] != 'A', '11'] = 0
test.loc[test['11'] == 'A', '11'] = 1

# Data normalization:
X = train.drop(['11'], axis=1)
sc = StandardScaler()

sc.fit(X)

X_1 = train.loc[train['11'] == 1.0]
X_0 = train.loc[train['11'] == 0.0]

X_1 = X_1.drop(['11'], axis=1)
columns = X_1.columns
ind = X_1.index
X_1 = sc.transform(X_1)
X_1 = pd.DataFrame(X_1, columns=columns, index=ind)

X_0 = X_0.drop(['11'], axis=1)
columns = X_0.columns
ind = X_0.index
X_0 = sc.transform(X_0)
X_0 = pd.DataFrame(X_0, columns=columns, index=ind)

Y_test = test[['11']]
X_test = test.drop(['11'], axis=1)
X_test = sc.transform(X_test)

Y_test = Y_test.to_numpy()

n = train.shape[0]
d = train.shape[1] - 1
number_of_estimations = 1
number_of_iterations = 1000
epsilon = 0.01
delta = 0.1

pi_0 = X_0.shape[0] / (X_0.shape[0] + X_1.shape[0])
pi_1 = 1 - pi_0

mu_1s = []
sigma_1s = []
mu_0s = []
sigma_0s = []

# Estimating Sigma and mu for number_of_estimation times.
for counter in range(number_of_estimations):
    # print(counter)
    mu, sigma = find_mean_and_covariance(X_1)
    mu_1s.append(mu)
    sigma_1s.append(sigma)

    mu, sigma = find_mean_and_covariance(X_0)
    mu_0s.append(mu)
    sigma_0s.append(sigma)

# Initializing w
w = 0.01 * np.ones(shape=(d, 1))

for iteration in range(number_of_iterations):

    if iteration % 100 == 99:
        print("Iteration number: ", iteration)

    # Initializing p_i s
    p0 = [0] * number_of_estimations
    p1 = [0] * number_of_estimations

    a0 = np.zeros(number_of_estimations)
    a1 = np.zeros(number_of_estimations)

    for i in range(number_of_estimations):
        current_data = np.random.multivariate_normal(mu_1s[i], sigma_1s[i], size=number_of_samples)

        res = - np.log(sigmoid(np.dot(current_data, w)))
        a1[i] = res.mean()

    for i in range(number_of_estimations):
        current_data = np.random.multivariate_normal(mu_0s[i], sigma_0s[i], size=number_of_samples)

        res = - np.log(1 - sigmoid(np.dot(current_data, w)))
        a0[i] = res.mean()

    sum_of_probabilities = sum(p1)
    lam_min = 0
    lam_max = max(a1)
    lam = 0

    j = 0
    t = 0
    while fabs(sum_of_probabilities - 1) > epsilon:

        lam = (lam_min + lam_max) / 2
        p1 = (a1 - lam) / (2 * delta)

        p1 = (p1 + np.fabs(p1)) / 2
        sum_of_probabilities = sum(p1)
        if sum_of_probabilities < 1:
            lam_max = lam

        else:
            lam_min = lam
        t += 1
        if t > 1000:
            print("Stuck!")

    lam_min = 0
    lam_max = max(a0)
    lam = 0
    sum_of_probabilities = sum(p0)
    j = 0
    t = 0
    while fabs(sum_of_probabilities - 1) > epsilon:

        lam = (lam_min + lam_max) / 2
        p0 = (a0 - lam) / (2 * delta)

        p0 = (p0 + np.fabs(p0)) / 2

        sum_of_probabilities = sum(p0)
        if sum_of_probabilities < 1:
            lam_max = lam

        else:
            lam_min = lam

        t += 1
        if t > 1000:
            print("Stuck!")

    # Updating w:
    total_grad = np.zeros(shape=(w.shape[0],))
    for counter in range(number_of_estimations):
        data_batch = np.random.multivariate_normal(mu_1s[counter], sigma_1s[counter], size=number_of_samples)

        inner = 1 - sigmoid(np.dot(data_batch, w))

        inner = np.diag(inner.flatten())
        grad = np.dot(inner, data_batch)

        grad = np.sum(grad, axis=0)
        total_grad -= p1[counter] * pi_1 * grad / number_of_samples

    for counter in range(number_of_estimations):
        data_batch = np.random.multivariate_normal(mu_0s[counter], sigma_0s[counter], size=number_of_samples)

        inner = sigmoid(np.dot(data_batch, w))

        inner = np.diag(inner.flatten())
        grad = np.dot(inner, data_batch)

        grad = np.sum(grad, axis=0)

        total_grad += p0[counter] * pi_0 * grad / number_of_samples

    total_grad = total_grad.reshape(w.shape)
    w -= 0.01 * total_grad

predictions = np.dot(X_test, w)
predictions = np.sign(predictions)

predictions = (predictions + 1) / 2
res = predictions == Y_test

count = np.sum(res, axis=0)
print(count[0] / res.shape[0])
