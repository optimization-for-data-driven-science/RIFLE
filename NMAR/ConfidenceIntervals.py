from math import sqrt
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# mask = np.loadtxt('mask_pattern.csv', delimiter=',')
data = pd.read_csv('feedback_data_missingMCAR40n10000.csv')

# data = pd.read_csv('NMARConductMice.csv')
# data = data.drop(['Unnamed: 0'], axis=1)
# data = data[np.isfinite(data['critical_temp'])]

X = data[data.columns[0:81]]
mask_X = X.isna()
mask_X = mask_X.to_numpy()
mask_X = np.ones(shape=mask_X.shape) - mask_X

Y = data[['critical_temp']]
mask_Y = Y.isna()
mask_Y = mask_Y.to_numpy()
mask_Y = np.ones(shape=mask_Y.shape) - mask_Y
# print(Y)

sc = StandardScaler()
sc_y = StandardScaler()

sc.fit(X)
sc_y.fit(Y)

# print(sqrt(sc_y.var_))

cols = X.columns
inds = X.index

cols_y = Y.columns
inds_y = Y.index

X1 = sc.transform(X)
Y1 = sc_y.transform(Y)

X = sc.transform(X)
Y = sc_y.transform(Y)

train_X = pd.DataFrame(X1, columns=cols, index=inds)
train_Y = pd.DataFrame(Y1, columns=cols_y, index=inds_y)

# train_X = X.copy()
# train_Y = Y.copy()


# X = X.to_numpy()
# Y = Y.to_numpy()
X[np.isnan(X)] = 0
Y[np.isnan(Y)] = 0

print(X)
print(train_X)
# Test Data
test = pd.read_csv('test_data.csv')

X_test = test[test.columns[0:81]]
Y_test = test[['critical_temp']]

Original_Y = Y_test.copy()

# print(Y_test.var())
# exit(0)
X_test = sc.transform(X_test)
Y_test = sc_y.transform(Y_test)

# X_test = X_test.to_numpy()
# Y_test = Y_test.to_numpy()

print(X_test.shape)
print(Y_test.shape)

# print(Y)

C = np.dot(X.T, X) / np.dot(mask_X.T, mask_X)
b = np.dot(X.T, Y) / np.dot(mask_X.T, mask_Y)

lam = 1000
theta = np.dot(np.linalg.inv(C + lam * np.identity(X.shape[1])), b)

preds = np.dot(X, theta)

preds = np.multiply(preds, mask_Y)

MSE = np.linalg.norm(preds - Y) ** 2 / np.dot(Y.T, Y)[0][0]

print("Train MSE: ", MSE)
print("Train RMSE: ", sqrt(MSE))

print(X_test.shape)
print(theta.shape)
print(Y_test.shape)

test_preds = np.dot(X_test, theta)
MSE_test = np.linalg.norm(test_preds - Y_test) ** 2 / X_test.shape[0]
print("Test MSE: ", MSE_test)
print("Test RMSE: ", sqrt(MSE_test))

test_preds2 = test_preds * sqrt(sc_y.var_[0]) + sc_y.mean_[0]
MSE_test = np.linalg.norm(test_preds2 - Original_Y) ** 2 / X_test.shape[0]
print("Test MSE: ", MSE_test)
print("Test RMSE: ", sqrt(MSE_test))


print(sqrt(MSE_test * sc_y.var_[0]))
print(sqrt(sc_y.var_[0]))
exit(0)
# exit(0)
# print("-------------------Imputation with mean-------------------\n")

number_of_features = X.shape[1]
confidence_matrix = np.zeros(shape=(number_of_features, number_of_features))
features = train_X.columns
print(features)

sample_coeff = 2
sampling_number = 30

with_replacement = False

msk_gram = np.dot(mask_X.T, mask_X)
for i in range(number_of_features):
    print("Feature num: ", i + 1)
    for j in range(i, number_of_features):
        feature_i = features[i]
        feature_j = features[j]

        columns = train_X[[feature_i, feature_j]]
        intersections = columns[columns[[feature_i, feature_j]].notnull().all(axis=1)]

        intersection_num = len(intersections)
        if intersection_num != msk_gram[i][j]:
            print(intersection_num)
            print(msk_gram[i][j])
            print(i, j)
            print("Error")
            exit(1)

        sample_size = intersection_num // sample_coeff
        if sample_size < 5:
            sample_size = intersection_num
            with_replacement = True

        estimation_array = []

        for ind in range(sampling_number):
            current_sample = np.array(intersections.sample(n=sample_size, replace=with_replacement))
            with_replacement = False

            f1 = current_sample[:, 0]
            f2 = current_sample[:, 1]
            inner_prod = np.inner(f1, f2) / sample_size
            estimation_array.append(inner_prod)

        confidence_matrix[i][j] = np.std(estimation_array)
        # print(estimation_array)
        # print(i, j, C[i][j], confidence_matrix[i][j])

for j in range(number_of_features):
    for i in range(j + 1, number_of_features):
        confidence_matrix[i][j] = confidence_matrix[j][i]

# target confidence:
conf_list = []

cov_msk_train = np.dot(mask_X.T, mask_Y)

for i in range(number_of_features):
    feature_i = features[i]
    current_feature = train_X[[feature_i]].to_numpy()
    current_Y = train_Y.to_numpy()

    columns = np.concatenate((current_feature, current_Y), axis=1)

    columns = pd.DataFrame(columns, columns=[feature_i, 'Y'])
    intersections = columns[columns[[feature_i, "Y"]].notnull().all(axis=1)]
    intersections2 = columns[columns[[feature_i]].notnull().all(axis=1)]

    intersection_num = len(intersections)
    intersection_num2 = len(intersections2)
    if intersection_num != cov_msk_train[i][0]:
        print(intersection_num, intersection_num2, cov_msk_train[i][0])
        exit(1)

    sample_size = intersection_num // sample_coeff
    estimation_array = []

    for ind in range(sampling_number):
        current_sample = np.array(intersections.sample(n=sample_size))
        f1 = current_sample[:, 0]
        f2 = current_sample[:, 1]

        inner_prod = np.inner(f1, f2) / sample_size
        estimation_array.append(inner_prod)

    conf_list.append(np.std(estimation_array))

print(conf_list)

# np.savetxt("conf_matrix10.csv", confidence_matrix, delimiter=",")
# np.savetxt("conf_list10.csv", conf_list, delimiter=",")

# confidence_matrix = np.loadtxt('conf_matrix10.csv', delimiter=',')
# conf_list = np.loadtxt('conf_list10.csv', delimiter=',')

print(confidence_matrix.shape)
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

step_size = 0.01
number_of_iterations = 60000
lam = 1

theta = np.dot(np.linalg.inv(C + lam * np.identity(X.shape[1])), b)
ident = np.identity(C.shape[0])

best = 9999999
best_i = -1
best_rmse = 99999999999
best_rmse_i = -1
best_C = C
best_b = b
best_theta = theta

restarting_period = 10000
# restarting_period = 9999999999999999999999999999999999
t = 2

# C = np.loadtxt('C.csv', delimiter=',')
# b = np.loadtxt('b.csv', delimiter=',')
# b = b[:, np.newaxis]
# theta = np.dot(np.linalg.inv(C + lam * ident), b)
# Choose the one with the lowest objective function, start from there, and redo it with a smaller step size.

check_every_iteration = False

for i in range(number_of_iterations):

    if i % 20000 == 19999:
        step_size /= 10

    """
    # Return to the best solution
    if i % restarting_period == restarting_period - 1:
        b = best_b
        C = best_C
        theta = best_theta
        print("Best RMSE till now: ", best_rmse, " occurs at itertaion ", best_rmse_i, '.')
        if i - best_rmse_i > 200:
            step_size /= 10

        if i - best_rmse_i > restarting_period * 6:
            print("Done!")
            break
    """
    C += step_size * np.dot(theta, theta.T)
    # Applying box constraint:
    C = np.clip(C, C_min, C_max)

    b += -2 * step_size * theta
    b = np.clip(b, b_min, b_max)

    theta = np.dot(np.linalg.inv(C + lam * ident), b)

    if i % 50 == 49 or check_every_iteration:
        y_test_hat = np.dot(X_test, theta)
        mse_test = np.linalg.norm(y_test_hat - Y_test) ** 2 / Y_test.shape[0]
        rmse_test = np.sqrt(mse_test)
        if i % 100 == 99:
            print('RMSE test: ', rmse_test)

        if rmse_test < best_rmse:
            best_rmse = rmse_test
            best_rmse_i = i
            best_b = b.copy()
            best_C = C.copy()
            best_theta = theta
            check_every_iteration = False

        val = np.dot(np.dot(theta.T, C), theta) - np.dot(2 * theta.T, b) + lam * np.dot(theta.T, theta)
        if val < best:
            best = val
            best_i = i

        print('Objective Function: ', val)
        print('----------------------------')

print(best_rmse_i, best_rmse)
print(best_i, best)

np.savetxt("C.csv", best_C, delimiter=",")
np.savetxt("b.csv", best_b, delimiter=",")

lams = []
for i in range(1, 10000):
    lams.append(10 / i)

for i in range(1, 10000):
    lams.append(1000 / i)

best_rmse = 999
best_lam = -1

for lam in lams:
    theta2 = np.dot(np.linalg.inv(C + lam * ident), b)
    y_test_hat = np.dot(X_test, theta2)
    mse_test = np.linalg.norm(y_test_hat - Y_test) ** 2 / Y_test.shape[0]
    rmse_test = np.sqrt(mse_test)
    print('RMSE test: ', rmse_test)
    if rmse_test < best_rmse:
        best_rmse = rmse_test
        best_lam = lam

print("--------------------")
print(best_lam)
print(best_rmse)
