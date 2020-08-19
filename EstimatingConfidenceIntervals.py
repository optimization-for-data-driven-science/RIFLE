import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import math


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
                     'DIQ050_1.0', 'DIQ050_2.0', 'DIQ275_0.0', 'DIQ275_1.0', 'DIQ080_0.0', 'DMDMARTL_1.0', 'DMDEDUC2', 'BPXPULS',
                     'BPQ030', 'BPQ080', 'BPQ070', 'DIQ175L', 'DIQ275', 'Y']
data = pd.read_csv("OGTT_Preprocessed20.csv")

data = data.loc[data['Y'].notnull()]

print(data)
# data = data[non_zero_features]
print(data.columns)

features = list(data.columns)
features = features[:-1]

scaler = StandardScaler()
df_normalized = scaler.fit_transform(data)

df_normalized = pd.DataFrame(df_normalized, index=data.index, columns=data.columns)

"""
train_index, test_index = get_test_index(df_normalized)
train = df_normalized.iloc[train_index]
test = df_normalized.iloc[test_index]
train2 = df_normalized.iloc[train2_index]

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

number_of_features = len(normalized_gram_train)
print(number_of_features)

# number_of_features = 10
confidence_matrix = np.zeros(shape=(number_of_features, number_of_features))

for i in range(number_of_features):
    print("Feature num: ", i + 1)
    for j in range(i, number_of_features):
        feature_i = features[i]
        feature_j = features[j]

        columns = train[[feature_i, feature_j]]

        intersections = columns[columns[[feature_i, feature_j]].notnull().all(axis=1)]

        intersection_num = len(intersections)
        if intersection_num != msk_gram_train[i][j]:
            print("Error")
            exit(1)

        sample_size = intersection_num // 10
        estimation_array = []

        for ind in range(10):
            current_sample = np.array(intersections.sample(n=sample_size))

            f1 = current_sample[:, 0]
            f2 = current_sample[:, 1]

            inner_prod = np.inner(f1, f2) / sample_size
            estimation_array.append(inner_prod)

        confidence_matrix[i][j] = np.std(estimation_array)

for j in range(number_of_features):
    for i in range(j + 1, number_of_features):
        confidence_matrix[i][j] = confidence_matrix[j][i]

print(confidence_matrix)
np.savetxt("confidences_non_missing.csv", confidence_matrix, delimiter=",")

# target confidence:
conf_list = []
for i in range(number_of_features):
    feature_i = features[i]
    columns = train[[feature_i, "Y"]]

    intersections = columns[columns[[feature_i, "Y"]].notnull().all(axis=1)]

    intersection_num = len(intersections)
    if intersection_num != cov_msk_train[i][0]:
        print("Error")
        exit(1)

    sample_size = intersection_num // 10
    estimation_array = []

    for ind in range(10):
        current_sample = np.array(intersections.sample(n=sample_size))
        f1 = current_sample[:, 0]
        f2 = current_sample[:, 1]

        inner_prod = np.inner(f1, f2) / sample_size
        estimation_array.append(inner_prod)

    conf_list.append(np.std(estimation_array))

print(conf_list)
np.savetxt("confidences_non_missing_y.csv", conf_list, delimiter=",")
