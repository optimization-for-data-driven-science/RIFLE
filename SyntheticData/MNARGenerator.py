import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from math import sqrt, fabs
import random
import scipy.stats as stat


def read_dataset(filename):
    data = pd.read_csv(filename, header=None)
    return data


def split_data(data):
    X = data.drop[data.columns[-1]]
    Y = data[[data.columns[-1]]]
    return X, Y


def sample_data(input_data, sample_num=500):
    return input_data.sample(sample_num)


def scale_training_and_test(train_data, test_data):
    sc = StandardScaler()
    sc.fit(train_data)

    train_scaled = sc.transform(train_data)
    test_scaled = sc.transform(test_data)

    return train_scaled, test_scaled


def generate_MNAR_data(data, q_low=0.5, q_high=1.0, scale=1.0, shift=0.0, is_scaled=False, cols=None, no_y=False):

    print("Shape of data: ", data.shape)
    original_data = data.copy()
    if not is_scaled:

        cols = data.columns
        sc = StandardScaler()
        data = sc.fit_transform(data)

    else:
        data = data.to_numpy()

    q_list = []
    number_of_features = data.shape[1]

    for i in range(number_of_features):
        q_list.append(random.uniform(q_low, q_high))

    if no_y:
        q_list[number_of_features - 1] = 0

    q_list[number_of_features - 1] = 0.5

    total = 0
    missing_count = 0

    mask_prob = np.zeros(shape=data.shape)
    mask = np.ones(shape=data.shape)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):

            total += 1

            val = scale * fabs(data[i][j]) + shift

            mask_prob[i][j] = stat.norm.cdf(val) * q_list[j]  # Probability of missing

            if random.random() < mask_prob[i][j]:
                mask[i][j] = 0
                missing_count += 1

    print("percentage of missing values: ", missing_count / total)
    # print(mask)

    mask[mask == 0] = 'nan'

    final_data = np.multiply(mask, original_data.to_numpy())

    pd_data = pd.DataFrame(final_data, columns=cols)
    return pd_data


X = np.loadtxt('X2000.csv', delimiter=',')

Y = np.loadtxt('Y2000.csv', delimiter=',')

Y = Y[:, np.newaxis]

data1 = np.concatenate((X, Y), axis=1)
df = pd.DataFrame(data1)


final = generate_MNAR_data(df, q_low=0, q_high=0.95, scale=2, shift=0, no_y=False)
print(final)
df.to_csv("Synthetic_original_MNAR_P40_2000.csv", index=None)
final.to_csv("Synthetic_missing_MNAR_P40_2000.csv", index=None)
