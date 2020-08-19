import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from math import sqrt


def write_xls(value, index, col_name, df):
    if col_name in df.columns:
        df.loc[index, col_name] = value
    return df


def save_to_xls(file, df):
    df.to_excel(file, index=False)


"""
train_data = pd.read_csv('MissForestImputed3000P95MCAR.csv')

print(train_data)
test_data = pd.read_csv('test_data.csv')

X = train_data[train_data.columns[0:-1]]
Y = train_data[[train_data.columns[-1]]]

Y_var = Y.var()[0]

print(X)
print(Y)

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
X_test = test_data[test_data.columns[0:-1]]

ratio = sqrt(Y_var) / sqrt(Y_test.var()[0])
test_scale = sqrt(Y_test.var()[0])

lam_list = [0.01, 0.1, 1, 2, 5, 10, 20, 50, 100, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 999999999999]

inds = X_test.index
cols = X_test.columns

X_test = sc.transform(X_test)
Y_test = sc_y.transform(Y_test)

X_test_df = pd.DataFrame(X_test, index=inds, columns=cols)

for lam in lam_list:
    ident = np.identity(X.shape[1])

    theta = np.dot(np.linalg.inv(np.dot(X.T, X) + lam * ident), np.dot(X.T, Y))

    print(X_test.shape)
    print(Y_test.shape)

    Y_hat = np.dot(X_test, theta)

    mse = np.linalg.norm(Y_test - Y_hat) ** 2 / Y_test.shape[0]
    print("Lambda: ", lam)
    # print("MissForest:")
    # print("Test MSE: ", mse)
    # print("Test RMSE: ", np.sqrt(mse))
    print("Corrected Test RMSE: ", np.sqrt(mse) * ratio)
"""
# -----------------------------------------------------------------------
"""
train_data = pd.read_csv('SuperConduct95n3000.csv')

drops = [
    'number_of_elements',
    'wtd_mean_atomic_mass',
    'gmean_atomic_mass',
    'range_atomic_mass',
    'wtd_range_atomic_mass',
    'std_atomic_mass',
    'wtd_std_atomic_mass',
    'wtd_mean_fie',
    'gmean_fie',
    'entropy_fie',
    'range_fie',
    'wtd_range_fie',
    'wtd_std_fie',
    'mean_atomic_radius',
    'gmean_atomic_radius',
    'wtd_range_atomic_radius',
    'gmean_Density',
    'wtd_gmean_Density',
    'std_Density',
    'wtd_std_Density',
    'gmean_ElectronAffinity',
    'wtd_entropy_ElectronAffinity',
    'std_ElectronAffinity',
    'wtd_std_ElectronAffinity',
    'gmean_FusionHeat',
    'entropy_FusionHeat',
    'wtd_entropy_FusionHeat',
    'wtd_std_FusionHeat',
    'gmean_ThermalConductivity',
    'entropy_ThermalConductivity',
    'wtd_entropy_ThermalConductivity',
    'range_ThermalConductivity',
    'wtd_range_ThermalConductivity',
    'mean_Valence',
    'wtd_entropy_Valence',
    'range_Valence',
    'wtd_range_Valence'
]
train_data = train_data.drop(['Unnamed: 0'], axis=1)

test_data = pd.read_csv('test_data.csv')
test_data = test_data.drop(drops, axis=1)

print(train_data.shape)
print(test_data.shape)
Y_test = test_data[[test_data.columns[-1]]]
X_test = test_data[test_data.columns[0:-1]]

print(X_test)

X = train_data[train_data.columns[0:-1]]
Y = train_data[[train_data.columns[-1]]]

ratio = sqrt(Y.var()) / sqrt(Y_test.var())
print(X)

sc = StandardScaler()
sc_y = StandardScaler()

sc.fit(X)
sc_y.fit(Y)

X = sc.transform(X)
Y = sc_y.transform(Y)

X_test = sc.transform(X_test)
Y_test = sc_y.transform(Y_test)
"""
"""
print("MICE:")

lam_list = [0.01, 0.1, 1, 2, 5, 10, 20, 50, 100, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 9999999999999999]
for lam in lam_list:
    # lam = 40000
    ident = np.identity(X.shape[1])

    theta = np.dot(np.linalg.inv(np.dot(X.T, X) + lam * ident), np.dot(X.T, Y))
    Y_hat = np.dot(X_test, theta)

    mse = np.linalg.norm(Y_test - Y_hat) ** 2 / Y_test.shape[0]
    print("Lam: ", lam)
    print("Test MSE: ", mse)
    print("Test RMSE: ", np.sqrt(mse))
    print("Corrected RMSE: ", np.sqrt(mse) * ratio)

"""
print("---------------------------")
print("Midas Prediction")

train_data = pd.read_csv('conductivity_MCAR_95_3000_noyimputed.csv')

train_data.drop(['Unnamed: 0'], axis=1, inplace=True)
print(train_data)
test_data = pd.read_csv('test_data.csv')
print(test_data)
X = train_data[train_data.columns[0:-1]]
Y = train_data[[train_data.columns[-1]]]

Y_test = test_data[[test_data.columns[-1]]]
X_test = test_data[test_data.columns[0:-1]]

ratio = sqrt(Y.var()) / sqrt(Y_test.var())
print(ratio)

sc = StandardScaler()
sc_y = StandardScaler()

sc.fit(X)
sc_y.fit(Y)

X = sc.transform(X)
Y = sc_y.transform(Y)

X_test = sc.transform(X_test)
Y_test = sc_y.transform(Y_test)


lam_list = [0.01, 0.1, 1, 2, 5, 10, 20, 50, 100, 500, 1000, 2000, 5000, 10000, 13000, 20000, 50000, 100000, 200000, 300000, 500000, 1000000, 2000000, 5000000]
excel_df = pd.read_excel('Test.xlsx')
for count, lam in enumerate(lam_list):
    ident = np.identity(X.shape[1])

    theta = np.dot(np.linalg.inv(np.dot(X.T, X) + lam * ident), np.dot(X.T, Y))
    Y_hat = np.dot(X_test, theta)

    mse = np.linalg.norm(Y_test - Y_hat) ** 2 / Y_test.shape[0]
    print("Lam: ", lam)
    print("Test MSE: ", mse)
    print("Test RMSE: ", np.sqrt(mse))
    print("Corrected RMSE: ", np.sqrt(mse) * ratio)
    excel_df = write_xls(np.sqrt(mse) * ratio, count, 'Midas', excel_df)
save_to_xls('Test2.xlsx', excel_df)
