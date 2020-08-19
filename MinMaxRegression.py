import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import math
from pandas import DataFrame
import scipy


def _getAplus(A):
    eigval, eigvec = scipy.linalg.eig(A)

    eigval = np.real(eigval)
    eigvec = np.real(eigvec)

    Q = np.matrix(eigvec)
    xdiag = np.matrix(np.diag(np.maximum(eigval, 0)))

    return Q * xdiag * Q.T


def get_test_index(df, test_size=0.10):
    train_data = df.iloc[:int((1 - test_size) * df.shape[0])]
    test_data = df.iloc[train_data.shape[0]:]
    return train_data.index, test_data[test_data["Y"].notnull()].index, test_data[test_data["Y"].isnull()].index


def get_test_index2(df, test_size=0.10):
    train_data = df.iloc[:int((1 - test_size) * df.shape[0])]
    test_data = df.iloc[train_data.shape[0]:]
    return train_data, test_data


non_zero_features = ['RIDAGEYR', 'BPXPLS', 'LBXGH', 'WTSAF2YR_x', 'LBXIN', 'PHAFSTHR', 'WTSAF2YR_y',
                     'LBXGLU', 'BPD035', 'DIQ230', 'DID250', 'DIQ300S', 'DID320', 'DIQ360', 'BMXWAIST',
                     'BMDAVSAD', 'BPXSY', 'DIDNEW260', 'RIDRETH3_4.0', 'RIDRETH3_6.0', 'RIDEXMON_1.0',
                     'DMDEDUC2_1.0', 'BPXPULS_2.0', 'BPQ020_1.0', 'BPQ020_2.0', 'BPQ030_2.0', 'BPQ040A_1.0',
                     'BPQ040A_2.0', 'BPQ080_1.0', 'BPQ080_2.0', 'BPQ060_1.0', 'BPQ060_2.0', 'BPQ070_1.0', 'BPQ090D_1.0',
                     'BPQ090D_2.0', 'DIQ172_1.0', 'DIQ175L_0.0', 'DIQ175L_1.0', 'DIQ175U_0.0', 'DIQ180_2.0',
                     'DIQ050_1.0', 'DIQ050_2.0', 'DIQ275_0.0', 'DIQ275_1.0', 'DIQ080_0.0', 'DMDMARTL_1.0', 'DMDEDUC2',
                     'BPXPULS',
                     'BPQ030', 'BPQ080', 'BPQ070', 'DIQ175L', 'DIQ275', 'Y']

data = pd.read_csv("OGTT_Preprocessed20.csv")
data = data.loc[data['Y'].notnull()]
# data = data[non_zero_features]

features = list(data.columns)
features = features[:-1]

scaler = StandardScaler()
df_normalized = scaler.fit_transform(data)

df_normalized = pd.DataFrame(df_normalized, index=data.index, columns=data.columns)

"""
train_index, test_index, train2_index = get_test_index(df_normalized)
train = df_normalized.iloc[train_index]
test = df_normalized.iloc[test_index]
train2 = df_normalized.iloc[train2_index]

train = pd.concat([train, train2], axis=0)
"""

train, test = get_test_index2(df_normalized)

Y_train = train["Y"].values
X_train = train.drop(["Y"], axis=1).values
Y_test = test["Y"].values
X_test = test.drop(["Y"], axis=1).values

Y_train = Y_train[:, np.newaxis]
Y_test = Y_test[:, np.newaxis]
print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)

# Compute Mask:
msk_X_train = np.isnan(X_train)
msk_X_train = np.ones(X_train.shape) - msk_X_train
msk_X_train = np.nan_to_num(msk_X_train)

msk_Y_train = np.isnan(Y_train)
msk_Y_train = np.ones(Y_train.shape) - msk_Y_train
msk_Y_train = np.nan_to_num(msk_Y_train)

X_train = np.nan_to_num(X_train)  # Set nan values to 0
Y_train = np.nan_to_num(Y_train)

print(X_train.shape)

# X^T X point estimation:
gram_train = np.dot(X_train.T, X_train)
msk_gram_train = np.dot(msk_X_train.T,
                        msk_X_train)  # Each entry of this matrix gives the number of intersections for the corresponding features

C = np.divide(gram_train, msk_gram_train)

cov_msk_train = np.dot(msk_X_train.T, msk_Y_train)
y_cov = np.dot(X_train.T, Y_train)
c = np.divide(y_cov, cov_msk_train)

confidences = pd.read_csv('confidences_non_missing.csv', header=None)
confidences = confidences.to_numpy()
print("-------------------")

const = 0.25
C_min = C - const * confidences
C_max = C + const * confidences

"""
for i in range(190):
    for j in range(i, 190):
        print("(", i, ',', j, '): ', C_min[i][j], ' < ', C[i][j], ' < ', C_max[i][j])

exit(0)
"""

y_confidences = pd.read_csv('confidences_non_missing_y.csv', header=None)
y_confidences = y_confidences.to_numpy()
print(y_confidences)
print(y_confidences.shape)

number_of_iterations = 500000
ident = np.identity(C.shape[0])
lam = 0.001

c_min = c - const * y_confidences
c_max = c + const * y_confidences

for i in range(len(y_cov)):
    print(c_min[i][0], c[i][0], c_max[i][0])

theta = np.zeros(shape=(C.shape[0], 1))

print("------------------------------------------------------------")
step_size = 0.0000001

step_size = 0.00001
for i in range(number_of_iterations):
    C += step_size * np.dot(theta, theta.T)
    # Applying box constraint:
    C = np.clip(C, C_min, C_max)

    # Applying PSD constraint:
    # C = _getAplus(C)

    #  temp = np.linalg.norm(C - C.T, 'fro')
    # print(temp)
    c += -2 * step_size * theta
    c = np.clip(c, c_min, c_max)

    theta = np.dot(np.linalg.inv(C + lam * ident), c)

# Applying Lasso to the updated C, c

"""
theta = np.zeros(shape=(C.shape[0], 1))
number_of_iterations = 7000
lambda1 = 0.07
lambda2 = 0.0  # Set zero if you don't want to have L_2 regulari
t_k = 0.001

shrinkage_parameter = lambda1 * t_k * np.ones(shape=theta.shape)
ones = np.ones(shape=theta.shape)

for i in range(number_of_iterations):
    grad = 2 * np.dot(C, theta) - 2 * c + 2 * lambda2 * theta
    shrinkage_input = theta - t_k * grad

    # Shrinkage
    temp = np.absolute(shrinkage_input) - shrinkage_parameter

    temp_sgn = (np.sign(temp) + ones) / 2
    val = np.multiply(temp, temp_sgn)
    theta = np.multiply(np.sign(shrinkage_input), val)

#  Compute MSE
y_hat_train = np.dot(X_train, theta)
y_hat_non_missing_train = np.multiply(msk_Y_train, y_hat_train)
y_non_missing_train = np.multiply(msk_Y_train, Y_train)
mse_train = np.linalg.norm(y_hat_non_missing_train - y_non_missing_train) ** 2 / np.sum(msk_Y_train)
rmse_train = np.sqrt(mse_train)
print("Train Error: ", rmse_train)

X_test = np.nan_to_num(X_test)
y_test_hat = np.dot(X_test, theta)
mse_test = np.linalg.norm(y_test_hat - Y_test) ** 2 / Y_test.shape[0]
rmse_test = np.sqrt(mse_test)
print("Test Error:", rmse_test)
"""


# theta = np.dot(np.linalg.inv(C + lam * ident), c)

y_hat_train = np.dot(X_train, theta)

y_hat_non_missing_train = np.multiply(msk_Y_train, y_hat_train)
y_non_missing_train = np.multiply(msk_Y_train, Y_train)

mse_train = np.linalg.norm(y_hat_non_missing_train - y_non_missing_train)**2/np.sum(msk_Y_train)

rmse_train = np.sqrt(mse_train)
print(rmse_train)

X_test = np.nan_to_num(X_test)
y_test_hat = np.dot(X_test, theta)

mse_test = np.linalg.norm(y_test_hat - Y_test)**2/Y_test.shape[0]
rmse_test = np.sqrt(mse_test)
print(rmse_test)

# print(theta)
