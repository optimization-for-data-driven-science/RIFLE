from math import sqrt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


predicts1 = []
predicts2 = []


train = pd.read_csv('train_missing_MCAR40_100_noy_1.csv')
test = pd.read_csv('avila-ts.csv')

test.loc[test['11'] != 'A', '11'] = 0
test.loc[test['11'] == 'A', '11'] = 1

X_test = test.drop(['11'], axis=1)
Y_test = test[['11']]

# Data normalization:
X = train.drop(['11'], axis=1)
sc = StandardScaler()

data_points = train.shape[0]
dimension = train.shape[1]

nulls = X.isnull().sum()

null_cols = []
columns2 = []
for i in range(len(nulls)):
    if nulls[i] >= X.shape[0]-1:
        columns2.append(X.columns[i])

X = X.drop(columns2, axis=1)

mask_X = X.isna()

mask_X = mask_X.to_numpy()
mask_X = np.ones(shape=mask_X.shape) - mask_X

Y = train[['11']]

train_std = sqrt(Y.var()[0])

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

Y_not_nan = np.nonzero(np.ones(Y.shape) - np.isnan(Y))[0]

np.isnan(Y)

X_test = sc.transform(X_test)

print(X_test)

train_X = pd.DataFrame(X1, columns=cols, index=inds)
train_Y = pd.DataFrame(Y1, columns=cols_y, index=inds_y)

X[np.isnan(X)] = 0
Y[np.isnan(Y)] = 0

mask_gram = np.dot(mask_X.T, mask_X)
mask_gram = np.where(mask_gram == 0, 1, mask_gram)

cov_gram = np.dot(mask_X.T, mask_Y)
cov_gram = np.where(cov_gram == 0, 1, cov_gram)

C = np.dot(X.T, X) / mask_gram
b = np.dot(X.T, Y) / cov_gram

number_of_features = X.shape[1]
confidence_matrix = np.zeros(shape=(number_of_features, number_of_features))
features = train_X.columns

sample_coeff = 1
sampling_number = 30

C_min = np.zeros(C.shape)
C_max = np.zeros(C.shape)

b_max = np.zeros(b.shape)
b_min = np.zeros(b.shape)

with_replacement = True
msk_gram = np.dot(mask_X.T, mask_X)
for i in range(number_of_features):
    print("Feature num: ", i + 1)
    for j in range(i, number_of_features):
        feature_i = features[i]
        feature_j = features[j]

        columns = train_X[[feature_i, feature_j]]
        # if i == 26:
        #     print(feature_i)
        #     print(columns)
        #     exit(0)
        intersections = columns[columns[[feature_i, feature_j]].notnull().all(axis=1)]

        intersection_num = len(intersections)
        if intersection_num != msk_gram[i][j]:
            print(intersection_num)
            print(msk_gram[i][j])
            print(i, j)
            print("Error")
            exit(1)

        sample_size = intersection_num // sample_coeff

        if sample_size < 2:
            max_vals = columns.max()
            max1 = max_vals[feature_i]
            max2 = max_vals[feature_j]
            C_max[i][j] = max1 * max2
            C_min[i][j] = - max1 * max2
            continue

        estimation_array = []

        for ind in range(sampling_number):
            current_sample = np.array(intersections.sample(n=sample_size, replace=with_replacement))
            with_replacement = False

            f1 = current_sample[:, 0]
            f2 = current_sample[:, 1]
            inner_prod = np.inner(f1, f2) / sample_size
            estimation_array.append(inner_prod)

        # confidence_matrix[i][j] = np.std(estimation_array)

        C_min[i][j] = min(estimation_array)
        C_max[i][j] = max(estimation_array)

        # print(estimation_array)
        # print(i, j, C[i][j], confidence_matrix[i][j])

for j in range(number_of_features):
    for i in range(j + 1, number_of_features):
        # confidence_matrix[i][j] = confidence_matrix[j][i]
        C_min[i][j] = C_min[j][i]
        C_max[i][j] = C_max[j][i]

# print("------------Confidence Matrix---------------")
# print(confidence_matrix)
# print("---------------------------")
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

    if sample_size < 2:
        max_vals = columns.max()
        max1 = max_vals[feature_i]
        max2 = max_vals["Y"]

        b_max[i][0] = max1 * max2
        b_min[i][0] = - max1 * max2
        # conf_list.append(max1 * max2)
        continue

    estimation_array = []

    for ind in range(sampling_number):
        current_sample = np.array(intersections.sample(n=sample_size, replace=with_replacement))
        f1 = current_sample[:, 0]
        f2 = current_sample[:, 1]

        inner_prod = np.inner(f1, f2) / sample_size
        estimation_array.append(inner_prod)

    # conf_list.append(np.std(estimation_array))
    b_max[i][0] = max(estimation_array)
    b_min[i][0] = min(estimation_array)

# print(conf_list)

# np.savetxt("conf_matrix.csv", confidence_matrix, delimiter=",")
# np.savetxt("conf_list.csv", conf_list, delimiter=",")

# confidence_matrix = np.loadtxt('conf_matrix.csv', delimiter=',')
# conf_list = np.loadtxt('conf_list.csv', delimiter=',')

# print(confidence_matrix.shape)
# print(conf_list)
# const = 1
# C_min = C - const * confidence_matrix
# C_max = C + const * confidence_matrix

# y_conf = np.asarray(conf_list)
# y_conf = y_conf[:, np.newaxis]
# print(y_conf.shape)
print(b.shape)

# b_min = b - const * y_conf
# b_max = b + const * y_conf

step_size = 0.001
number_of_iterations = 100000
lam = 0.00001
print("----------------------")


predicts = []

ident = np.identity(C.shape[0])

theta = np.zeros(shape=(X.shape[1], 1))
theta2 = np.dot(np.linalg.inv(C + lam * ident), b)

for k in range(number_of_iterations):
    C += step_size * np.dot(theta, theta.T)
    C = np.clip(C, C_min, C_max)

    b += -2 * step_size * theta
    b = np.clip(b, b_min, b_max)

    theta = np.dot(np.linalg.inv(C + lam * ident), b)


Y_pred = np.dot(X_test, theta2)
Y_pred = Y_pred * sc_y.scale_[0] + sc_y.mean_[0]

print(Y_pred)
print(Y_pred.shape)

print(Y_pred - 0.5)

Y_predictions = (np.sign(Y_pred - 0.5) + 1) / 2
print(Y_predictions.shape)

Y_test = Y_test.to_numpy()
print(Y_test.shape)

res = Y_predictions == Y_test

count = np.sum(res, axis=0)
print(count[0] / res.shape[0])
