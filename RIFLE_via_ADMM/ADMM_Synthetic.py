from math import sqrt, fabs
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from numpy import genfromtxt


# Grid configurations
plt.grid(True, linewidth=0.2, c='k')


def get_near_psd(A_matrix):
    A_sym = (A_matrix + A_matrix.T) / 2
    eigval, eigvec = np.linalg.eigh(A_sym)
    eigval[eigval < 0] = 0

    return np.dot(eigvec, np.dot(np.diag(eigval), eigvec.T))


def is_pos_semidef(x):
    return np.all(np.linalg.eigvals(x) >= 0)


data = pd.read_csv('../SyntheticData/data_mcar40_d50_1K.csv')
d = str(50)

X_test = np.loadtxt('X1K_d50_test.csv', delimiter=',')
Y_test = np.loadtxt('Y1K_d50_test.csv', delimiter=',')
Y_test = Y_test[:, np.newaxis]

X = data[data.columns[0:-1]]
Y = data[[d]]

MSK_X = X.isna()
MSK_Y = Y.isna()


MSK_X = MSK_X.values
MSK_Y = MSK_Y.values
originalX = X.copy()
originalY = Y.copy()

# X = pd.DataFrame(X, index=originalX.index, columns=originalX.columns)
# Y = pd.DataFrame(Y)

Y['target'] = Y['50']
Y = Y.drop(['50'], axis=1)

number_of_test_points = Y_test.shape[0]
original_std = np.nanstd(Y_test)

data_points = X.shape[0]

mask_X = X.isna()

mask_X = mask_X.to_numpy()
mask_X = np.ones(shape=mask_X.shape) - mask_X

train_std = sqrt(Y.var()[0])
train_mean = Y.mean()[0]

mask_Y_test = Y.isna()
mask_Y_test = mask_Y_test.to_numpy()
missing_entries = mask_Y_test.sum(axis=0)[0]
mask_Y = np.ones(shape=mask_Y_test.shape) - mask_Y_test

sc = StandardScaler()
sc_y = StandardScaler()

sc.fit(X)
sc_y.fit(Y)

cols = X.columns
inds = X.index

cols_y = Y.columns
inds_y = Y.index

X1 = sc.transform(X)
Y1 = sc_y.transform(Y)

X = sc.transform(X)
Y = sc_y.transform(Y)

X_test = sc.transform(X_test)

train_X = pd.DataFrame(X1, columns=cols, index=inds)
train_Y = pd.DataFrame(Y1, columns=cols_y, index=inds_y)

X[np.isnan(X)] = 0
Y[np.isnan(Y)] = 0

C = np.dot(X.T, X) / np.dot(mask_X.T, mask_X)
b = np.dot(X.T, Y) / np.dot(mask_X.T, mask_Y)
lam = 1
theta = np.dot(np.linalg.inv(C + lam * np.identity(X.shape[1])), b)

Y_pred = np.dot(X_test, theta)
Y_pred = train_std * Y_pred + train_mean

mse = np.linalg.norm(Y_pred - Y_test) ** 2 / number_of_test_points
print("RMSE: ", sqrt(mse))
print("Scaled: ", sqrt(mse) / original_std)
print("-----------------------------")

number_of_features = X.shape[1]
confidence_matrix = np.zeros(shape=(number_of_features, number_of_features))
features = train_X.columns
print(features)

sample_coeff = 100
sampling_number = 30

with_replacement = True

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

print("------------Confidence Matrix---------------")
print(confidence_matrix)
print("---------------------------")
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
        current_sample = np.array(intersections.sample(n=sample_size, replace=with_replacement))
        f1 = current_sample[:, 0]
        f2 = current_sample[:, 1]

        inner_prod = np.inner(f1, f2) / sample_size
        estimation_array.append(inner_prod)

    conf_list.append(np.std(estimation_array))

print(conf_list)

np.savetxt("conf_matrix.csv", confidence_matrix, delimiter=",")
np.savetxt("conf_list.csv", conf_list, delimiter=",")

# confidence_matrix = np.loadtxt('conf_matrix.csv', delimiter=',')
# conf_list = np.loadtxt('conf_list.csv', delimiter=',')

# print(confidence_matrix.shape)
# print(conf_list)
const = 0.1
C_min = C - const * confidence_matrix
C_max = C + const * confidence_matrix

y_conf = np.asarray(conf_list)
y_conf = y_conf[:, np.newaxis]

b_min = b - const * y_conf
b_max = b + const * y_conf

sample_coeff = 0.01
sampling_number = 30

number_of_iterations = 2000

rho_list = [0.5, 1]

plots = []
print("----------------------")

checker = False

min_cost = 999999999999999999999999
optimal_theta = np.zeros(theta.shape)

rho = 0.05
b_init = np.dot(X.T, Y) / np.dot(mask_X.T, mask_Y)
C_init = np.dot(X.T, X) / np.dot(mask_X.T, mask_X)
theta_init = np.dot(np.linalg.inv(C + lam * np.identity(X.shape[1])), b)
# rho = 10

d_prime_init = np.zeros(theta.shape)
e_prime_init = np.zeros(theta.shape)
theta_prime_init = theta.copy()
mu_d_init = np.zeros(theta.shape)
mu_e_init = np.zeros(theta.shape)
mu_theta_init = np.zeros(theta.shape)
eta_init = np.zeros(theta.shape)

A_init = np.zeros(C.shape)
A_prime_init = np.zeros(C.shape)
B_init = np.zeros(C.shape)
B_prime_init = np.zeros(C.shape)
G_init = np.zeros(C.shape)
M_A_init = np.zeros(C.shape)
M_B_init = np.zeros(C.shape)
Gamma_init = np.zeros(C.shape)

rmse_list = []

for _ in range(12):

    # First block (theta, d, e, A_prime, B_prime, G)
    # Updating theta, d, e
    c_1 = b_min + rho * d_prime_init - mu_d_init + eta_init
    c_2 = -b_max + rho * e_prime_init - mu_e_init - eta_init
    c_3 = rho * theta_prime_init - mu_theta_init - 2 * eta_init
    theta_init = 1 / (6 * lam + 7 * rho) * (2 * c_1 - 2 * c_2 + 3 * c_3)
    d_init = 1 / (6 * lam + 7 * rho) * ((6 * rho + 4 * lam) / rho * c_1 + (rho + 4 * lam) / rho * c_2 + 2 * c_3)
    e_init = 1 / (6 * lam + 7 * rho) * ((rho + 2 * lam) / rho * c_1 + (6 * rho + 2 * lam) / rho * c_2 - 2 * c_3)

    # Updating A_prime, B_prime, G
    A_prime_init = np.maximum(A_init + M_A_init / rho, 0)
    B_prime_init = np.maximum(B_init + M_B_init / rho, 0)
    G_init = get_near_psd(B_init - A_init + Gamma_init / rho - np.dot(theta_prime_init, theta_prime_init.T)) + np.dot(theta_prime_init, theta_prime_init.T)
    # G = B - A + Gamma / rho

    # Second Block (d_prime, e_prime, theta_prime, A, B)

    # Updating d_prime, e_prime
    d_prime = np.maximum(d_init + mu_d_init / rho, 0)
    e_prime = np.maximum(e_init + mu_e_init / rho, 0)
    # theta_prime = theta + mu_theta / rho

    # Updating A and B
    D_1 = rho * A_prime_init - rho * G_init + Gamma_init - M_A_init + C_min
    D_2 = rho * B_prime_init + rho * G_init - Gamma_init - M_B_init - C_max
    A_init = 1 / (3 * rho) * (2 * D_1 + D_2)
    B_init = 1 / (3 * rho) * (D_1 + 2 * D_2)

    # symmetric_G = (G + G.T) / 2
    # print(symmetric_G - G)

    # Updating theta_prime
    alpha = theta_init + mu_theta_init / rho
    U, S, VT = np.linalg.svd(G_init)

    gamma = np.dot(U.T, alpha)

    optimal_solution = np.zeros(shape=S.shape)
    mu_star_min = 0
    mu_star_max = 10e5
    current_mu = (mu_star_min + mu_star_max) / 2

    summation = 0
    for ind in range(optimal_solution.shape[0]):
        optimal_solution[ind] = S[ind] * gamma[ind][0] / (S[ind] + current_mu)
        summation += optimal_solution[ind] * optimal_solution[ind] / (S[ind] + 0.00001)

    max_iter = 0

    while (summation - 1 > 0.01 or summation - 1 < - 0.01) and max_iter < 5000:
        max_iter += 1
        if summation - 1 > 0.01:
            mu_star_min = current_mu

        else:
            mu_star_max = current_mu

        current_mu = (mu_star_min + mu_star_max) / 2
        summation = 0
        for ind in range(optimal_solution.shape[0]):
            optimal_solution[ind] = S[ind] * gamma[ind][0] / (S[ind] + current_mu)
            summation += optimal_solution[ind] * optimal_solution[ind] / S[ind]

    optimal_solution = optimal_solution[:, np.newaxis]
    theta_prime_init = np.dot(U, optimal_solution)
    # Updating Multipliers
    mu_d_init += rho * (d_init - d_prime_init)
    mu_e_init += rho * (e_init - e_prime_init)
    mu_theta_init += rho * (theta_init - theta_prime_init)
    eta_init += rho * (2 * theta_init - d_init + e_init)
    M_A_init += rho * (A_init - A_prime_init)
    M_B_init += rho * (B_init - B_prime_init)
    Gamma_init += rho * (B_init - A_init - G_init)


# Main Part
for rho in rho_list:

    cost_array = []
    real_cost_array = []
    constraint_array = []

    b = b_init.copy()
    C = C_init.copy()
    theta = theta_init.copy()
    # rho = 10

    d_prime = d_prime_init.copy()
    e_prime = e_prime_init.copy()
    theta_prime = theta_prime_init.copy()
    mu_d = mu_d_init.copy()
    mu_e = mu_e_init.copy()
    mu_theta = mu_theta_init.copy()
    eta = eta_init.copy()

    A = A_init.copy()
    A_prime = A_prime_init.copy()
    B = B_init.copy()
    B_prime = B_prime_init.copy()
    G = G_init.copy()
    M_A = M_A_init.copy()
    M_B = M_B_init.copy()
    Gamma = Gamma_init.copy()

    for j in range(number_of_iterations):

        # First block (theta, d, e, A_prime, B_prime, G)
        # Updating theta, d, e
        c_1 = b_min + rho * d_prime - mu_d + eta
        c_2 = -b_max + rho * e_prime - mu_e - eta
        c_3 = rho * theta_prime - mu_theta - 2 * eta
        theta = 1 / (6 * lam + 7 * rho) * (2 * c_1 - 2 * c_2 + 3 * c_3)
        d = 1 / (6 * lam + 7 * rho) * ((6 * rho + 4 * lam) / rho * c_1 + (rho + 4 * lam) / rho * c_2 + 2 * c_3)
        e = 1 / (6 * lam + 7 * rho) * ((rho + 2 * lam) / rho * c_1 + (6 * rho + 2 * lam) / rho * c_2 - 2 * c_3)

        # Updating A_prime, B_prime, G
        A_prime = np.maximum(A + M_A / rho, 0)
        B_prime = np.maximum(B + M_B / rho, 0)
        G = get_near_psd(B - A + Gamma / rho - np.dot(theta_prime, theta_prime.T)) + np.dot(theta_prime, theta_prime.T)
        # G = B - A + Gamma / rho

        # Second Block (d_prime, e_prime, theta_prime, A, B)

        # Updating d_prime, e_prime
        d_prime = np.maximum(d + mu_d / rho, 0)
        e_prime = np.maximum(e + mu_e / rho, 0)
        # theta_prime = theta + mu_theta / rho

        # Updating A and B
        D_1 = rho * A_prime - rho * G + Gamma - M_A + C_min
        D_2 = rho * B_prime + rho * G - Gamma - M_B - C_max
        A = 1 / (3 * rho) * (2 * D_1 + D_2)
        B = 1 / (3 * rho) * (D_1 + 2 * D_2)

        # symmetric_G = (G + G.T) / 2
        # print(symmetric_G - G)

        # Updating theta_prime
        alpha = theta + mu_theta / rho
        U, S, VT = np.linalg.svd(G)

        gamma = np.dot(U.T, alpha)

        optimal_solution = np.zeros(shape=S.shape)
        mu_star_min = 0
        mu_star_max = 10e5
        current_mu = (mu_star_min + mu_star_max) / 2

        summation = 0
        for ind in range(optimal_solution.shape[0]):
            optimal_solution[ind] = S[ind] * gamma[ind][0] / (S[ind] + current_mu)
            summation += optimal_solution[ind] * optimal_solution[ind] / (S[ind] + 0.00001)

        max_iter = 0

        while (summation - 1 > 0.01 or summation - 1 < - 0.01) and max_iter < 5000:
            max_iter += 1
            if summation - 1 > 0.01:
                # current mu should be bigger
                mu_star_min = current_mu

            else:
                mu_star_max = current_mu

            current_mu = (mu_star_min + mu_star_max) / 2
            summation = 0
            for ind in range(optimal_solution.shape[0]):
                optimal_solution[ind] = S[ind] * gamma[ind][0] / (S[ind] + current_mu)
                summation += optimal_solution[ind] * optimal_solution[ind] / S[ind]

        optimal_solution = optimal_solution[:, np.newaxis]
        theta_prime = np.dot(U, optimal_solution)
        # Updating Multipliers
        mu_d += rho * (d - d_prime)
        mu_e += rho * (e - e_prime)
        mu_theta += rho * (theta - theta_prime)
        eta += rho * (2 * theta - d + e)
        M_A += rho * (A - A_prime)
        M_B += rho * (B - B_prime)
        Gamma += rho * (B - A - G)

        # Calculating The objective function:
        cost = - np.dot(b_min.T, d)[0][0] + np.dot(b_max.T, e)[0][0] - np.sum(np.multiply(C_min, A)) + \
               np.sum(np.multiply(C_max, B)) + lam * np.linalg.norm(theta) ** 2 + \
               np.sum(np.multiply(A - A_prime, M_A)) + \
               rho / 2 * np.linalg.norm(A - A_prime, 'fro') ** 2 + np.sum(np.multiply(B - B_prime, M_B)) + \
               rho / 2 * np.linalg.norm(B - B_prime, 'fro') ** 2 + np.dot(mu_d.T, d - d_prime)[0][0] + \
               rho / 2 * np.linalg.norm(d - d_prime) ** 2 + np.dot(mu_e.T, e - e_prime)[0][0] + \
               rho / 2 * np.linalg.norm(e - e_prime) ** 2 + np.dot(mu_theta.T, theta - theta_prime)[0][0] + \
               rho / 2 * np.linalg.norm(theta - theta_prime) ** 2 + np.dot(eta.T, 2 * theta - d + e)[0][0] + \
               rho / 2 * np.linalg.norm(2 * theta - d + e) ** 2 + np.sum(np.multiply(B - A - G, Gamma)) + \
               rho / 2 * np.linalg.norm(B - A - G, 'fro') ** 2

        real_cost = - np.dot(b_min.T, d)[0][0] + np.dot(b_max.T, e)[0][0] - np.sum(np.multiply(C_min, A)) + np.sum(
            np.multiply(C_max, B)) + lam * np.linalg.norm(theta) ** 2
        print(cost, real_cost)
        """
        print("||d - d'||^2: ", np.linalg.norm(d - d_prime) ** 2)
        print("||e - e'||^2: ", np.linalg.norm(e - e_prime) ** 2)
        print("||theta - theta'||^2: ",
              np.linalg.norm(theta - theta_prime) ** 2 / (np.linalg.norm(theta) ** 2 + np.linalg.norm(theta_prime) ** 2))
        print("||2theta - d + e||^2: ", np.linalg.norm(2 * theta - d + e) ** 2)
        print("||A - A'||^2: ", np.linalg.norm(A - A_prime, 'fro') ** 2)
        print("||B - B'||^2: ", np.linalg.norm(B - B_prime, 'fro') ** 2)
        print("||B - A - G||^2: ", np.linalg.norm(B - A - G, 'fro') ** 2)
        print(- np.dot(b_min.T, d)[0][0])
        print(np.dot(b_max.T, e)[0][0])
        print(- np.sum(np.multiply(C_min, A)))
        print(np.sum(np.multiply(C_max, B)))
        print(lam * np.linalg.norm(theta) ** 2)
        print('A norm: ', np.linalg.norm(A, 'fro') ** 2)
        print('B norm: ', np.linalg.norm(B, 'fro') ** 2)
        print('d norm: ', np.linalg.norm(d) ** 2)
        print('e norm: ', np.linalg.norm(e) ** 2)
        print('theta norm: ', np.linalg.norm(theta) ** 2)
        print("#################")
        """
        if not checker:
            real_cost_array.append(real_cost)
            cost_array.append(cost)
            constraint_array.append(fabs(cost - real_cost))

            # print(real_cost)
        checker = False

        if real_cost < min_cost and j > 100:
            min_cost = real_cost
            optimal_theta = theta.copy()

    print("||d - d'||^2: ", np.linalg.norm(d - d_prime) ** 2)
    print("||e - e'||^2: ", np.linalg.norm(e - e_prime) ** 2)
    print("||theta - theta'||^2: ",
          np.linalg.norm(theta - theta_prime) ** 2 / (np.linalg.norm(theta) ** 2 + np.linalg.norm(theta_prime) ** 2))
    print("||2theta - d + e||^2: ", np.linalg.norm(2 * theta - d + e) ** 2)
    print("||A - A'||^2: ", np.linalg.norm(A - A_prime, 'fro') ** 2)
    print("||B - B'||^2: ", np.linalg.norm(B - B_prime, 'fro') ** 2)
    print("||B - A - G||^2: ", np.linalg.norm(B - A - G, 'fro') ** 2)
    print(- np.dot(b_min.T, d)[0][0])
    print(np.dot(b_max.T, e)[0][0])
    print(- np.sum(np.multiply(C_min, A)))
    print(np.sum(np.multiply(C_max, B)))
    print(lam * np.linalg.norm(theta) ** 2)
    print('A norm: ', np.linalg.norm(A, 'fro') ** 2)
    print('B norm: ', np.linalg.norm(B, 'fro') ** 2)
    print('d norm: ', np.linalg.norm(d) ** 2)
    print('e norm: ', np.linalg.norm(e) ** 2)
    print('theta norm: ', np.linalg.norm(theta) ** 2)
    print("G - theta' theta'^T", is_pos_semidef(G - np.dot(theta_prime, theta_prime.T)))
    H = np.linalg.eigvals(G - np.dot(theta, theta.T))
    print(H)
    Y_pred = np.dot(X_test, theta)
    Y_pred = train_std * Y_pred + train_mean
    mse = np.linalg.norm(Y_pred - Y_test) ** 2 / number_of_test_points
    print("RMSE: ", sqrt(mse))
    print("Scaled: ", sqrt(mse) / original_std)
    rmse_list.append(sqrt(mse) / original_std)
    print("-----------------------------")

    """
    print("And now optimal:")
    print("Min cost: ", min_cost)
    Y_pred = np.dot(X_test, optimal_theta)
    Y_pred = train_std * Y_pred + train_mean
    mse = np.linalg.norm(Y_pred - Y_test) ** 2 / number_of_test_points
    print("RMSE: ", sqrt(mse))
    print("Scaled: ", sqrt(mse) / original_std)
    """

    plots.append(real_cost_array)


for item in plots:
    plt.plot(item)

plt.show()

print(rmse_list)
