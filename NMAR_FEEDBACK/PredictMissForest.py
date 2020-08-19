import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from math import sqrt

train_data = pd.read_csv('MissForestImputedNMAR50n1K.csv')

print(train_data.shape)

test_data = pd.read_csv('test_data.csv')

X = train_data[train_data.columns[0:280]]
Y = train_data[[train_data.columns[-1]]]

Y_var = Y.var()[0]

print(X)
print(Y)
print(Y.shape)

sc = StandardScaler()
sc_y = StandardScaler()

sc.fit(X)
sc_y.fit(Y)

X = sc.transform(X)
Y = sc_y.transform(Y)

var = sc_y.var_[0]
scale = sqrt(var)

# X = X.to_numpy()
# Y = Y.to_numpy()

print(X.shape)
print(Y.shape)

Y_test = test_data[[test_data.columns[-1]]]
X_test = test_data[test_data.columns[0:280]]

ratio = sqrt(Y_var) / sqrt(Y_test.var()[0])

lam = 1140
ident = np.identity(X.shape[1])

theta = np.dot(np.linalg.inv(np.dot(X.T, X) + lam * ident), np.dot(X.T, Y))

test_scale = sqrt(Y_test.var()[0])
# X_test = sc.transform(X_test)
# Y_test = sc_y.transform(Y_test)

inds = X_test.index
cols = X_test.columns

X_test = sc.transform(X_test)
Y_test = sc_y.transform(Y_test)

X_test_df = pd.DataFrame(X_test, index=inds, columns=cols)

print(X_test.shape)
print(Y_test.shape)

Y_hat = np.dot(X_test, theta)

mse = np.linalg.norm(Y_test - Y_hat) ** 2 / Y_test.shape[0]

print("MissForest:")
print("Test MSE: ", mse)
print("Test RMSE: ", np.sqrt(mse))
print("Corrected Test RMSE: ", np.sqrt(mse) * ratio)
exit(0)
# -----------------------------------------------------------------------
train_data = pd.read_csv('FeedbackMICE50n1K.csv')

test_data = pd.read_csv('test_data.csv')

Y_test = test_data[[test_data.columns[-1]]]
X_test = test_data[test_data.columns[0:280]]

drop_cols = [
    13,
    18,
    22,
    23,
    43,
    100,
    179,
    209,
    225,
    227,
    266,
    272,
    273,
    7,
    12,
    17,
    27,
    31,
    32,
    37,
    38,
    40,
    42,
    49,
    58,
    62,
    64,
    65,
    69,
    70,
    72,
    73,
    74,
    75,
    77,
    79,
    80,
    81,
    82,
    83,
    84,
    86,
    89,
    90,
    91,
    92,
    93,
    94,
    96,
    97,
    98,
    99,
    105,
    106,
    108,
    110,
    111,
    112,
    117,
    123,
    125,
    126,
    128,
    129,
    131,
    136,
    141,
    147,
    148,
    149,
    151,
    153,
    154,
    155,
    159,
    160,
    163,
    164,
    165,
    166,
    167,
    168,
    170,
    171,
    172,
    174,
    175,
    177,
    178,
    180,
    184,
    185,
    189,
    195,
    197,
    198,
    199,
    200,
    202,
    205,
    206,
    210,
    211,
    213,
    214,
    216,
    217,
    219,
    221,
    222,
    223,
    224,
    226,
    228,
    230,
    234,
    237,
    238,
    239,
    240,
    242,
    243,
    249,
    251,
    252,
    255,
    257,
    258,
    261,
    277,
    278]

train_data = train_data.drop(['Unnamed: 0'], axis=1)


X = train_data[train_data.columns[0:149]]
Y = train_data[[train_data.columns[-1]]]

ratio = sqrt(Y.var()) / sqrt(Y_test.var())
print(X)
X_test = X_test.drop(X_test.columns[drop_cols], axis=1)
# X_test_df = X_test_df.drop(X_test_df.columns[-1], axis=1)


sc = StandardScaler()
sc_y = StandardScaler()

sc.fit(X)
sc_y.fit(Y)


X = sc.transform(X)
Y = sc_y.transform(Y)

X_test = sc.transform(X_test)
Y_test = sc_y.transform(Y_test)

"""
X = X.to_numpy()
Y = Y.to_numpy()

X_test = X_test.to_numpy()
Y_test = Y_test.to_numpy()
"""

lam = 9500
ident = np.identity(X.shape[1])

theta = np.dot(np.linalg.inv(np.dot(X.T, X) + lam * ident), np.dot(X.T, Y))
Y_hat = np.dot(X_test, theta)

mse = np.linalg.norm(Y_test - Y_hat) ** 2 / Y_test.shape[0]

print("MICE:")
print("Test MSE: ", mse)
print("Test RMSE: ", np.sqrt(mse))
print("Corrected RMSE: ", np.sqrt(mse) * ratio)
exit(0)
