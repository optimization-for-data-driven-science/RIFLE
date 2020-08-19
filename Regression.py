import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def get_test_index(df, test_size=0.15):
    train_data = df.iloc[:int((1 - test_size) * df.shape[0])]
    test_data = df.iloc[train_data.shape[0]:]
    return train_data.index, test_data[test_data["Y"].notnull()].index, test_data[test_data["Y"].isnull()].index


def get_test_index2(df, test_size=0.10):
    train_data = df.iloc[:int((1 - test_size) * df.shape[0])]
    test_data = df.iloc[train_data.shape[0]:]
    return train_data, test_data


data = pd.read_csv("OGTT_Preprocessed20.csv")
data = data.loc[data['Y'].notnull()]

scaler = StandardScaler()
df_normalized = scaler.fit_transform(data)

df_normalized = pd.DataFrame(df_normalized, index=data.index, columns=data.columns)

# scaler = StandardScaler()
# df_normalized = scaler.fit_transform(data)

# df_normalized = pd.DataFrame(df_normalized, index=data.index, columns=data.columns)
"""
train_index, test_index, train2_index = get_test_index(data)
train = data.iloc[train_index]
test = data.iloc[test_index]
train2 = data.iloc[train2_index]

train = pd.concat([train, train2], axis=0)
"""

train, test = get_test_index2(df_normalized)
print(train.shape)
print(test.shape)

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

# X^T X estimation:
gram_train = np.dot(X_train.T, X_train)
msk_gram_train = np.dot(msk_X_train.T,
                        msk_X_train)  # Each entry of this matrix gives the number of intersections for the corresponding features

intersections = []
for i in range(len(msk_gram_train)):
    for j in range(len(msk_gram_train)):
        intersections.append(msk_gram_train[i][j])

intersections = np.sort(intersections)
print(intersections)

normalized_gram_train = np.divide(gram_train, msk_gram_train)

cov_msk_train = np.dot(msk_X_train.T, msk_Y_train)
cov_train = np.dot(X_train.T, Y_train)
print(cov_train)
cov_train = np.divide(cov_train, cov_msk_train)

row_idx = np.where(cov_train == 0)[0]
print(row_idx)
print(data.columns[list(row_idx)])

# Ridge Regression
lambda_prime = 1
ident = np.identity(normalized_gram_train.shape[0])
ident = lambda_prime * ident
ident[0, 0] = 0  # Bias term

theta_train = np.dot(np.linalg.inv(normalized_gram_train + ident), cov_train)

# Compute MSE
y_hat_train = np.dot(X_train, theta_train)

y_hat_non_missing_train = np.multiply(msk_Y_train, y_hat_train)
y_non_missing_train = np.multiply(msk_Y_train, Y_train)

mse_train = np.linalg.norm(y_hat_non_missing_train - y_non_missing_train) ** 2 / np.sum(msk_Y_train)

rmse_train = np.sqrt(mse_train)
print(rmse_train)

X_test = np.nan_to_num(X_test)
y_test_hat = np.dot(X_test, theta_train)

mse_test = np.linalg.norm(y_test_hat - Y_test) ** 2 / Y_test.shape[0]
rmse_test = np.sqrt(mse_test)
print(rmse_test)
