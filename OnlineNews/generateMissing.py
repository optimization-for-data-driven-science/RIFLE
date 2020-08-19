import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from math import sqrt, fabs
import random
import scipy.stats as stat


def read_dataset(filename):
    data = pd.read_csv(filename)
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


def generate_MNAR_data(data, q_low=0.5, q_high=1.0, scale=1.0, shift=0.0, is_scaled=False, cols=None, no_y=False,
                       binary_variables=None):
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

    total = 0
    missing_count = 0

    mask_prob = np.zeros(shape=data.shape)
    mask = np.ones(shape=data.shape)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            """
            if j in binary_variables:
                mask_prob[i][j] = q_list[j]
            """

            val = scale * fabs(data[i][j]) - shift

            mask_prob[i][j] = stat.norm.cdf(val) * q_list[j]  # Probability of missing

            if random.random() < mask_prob[i][j]:
                mask[i][j] = 0
                missing_count += 1

            total += 1

    print("percentage of missing values: ", missing_count / total)
    print(mask)

    mask[mask == 0] = 'nan'

    final_data = np.multiply(mask, original_data.to_numpy())

    pd_data = pd.DataFrame(final_data, columns=cols)
    return pd_data

"""
binary_vars = ['data_channel_is_lifestyle',
               'data_channel_is_entertainment',
               'data_channel_is_bus',
               'data_channel_is_socmed',
               'data_channel_is_tech',
               'data_channel_is_world',
               'weekday_is_monday',
               'weekday_is_tuesday',
               'weekday_is_wednesday',
               'weekday_is_thursday',
               'weekday_is_friday',
               'weekday_is_saturday',
               'weekday_is_sunday',
               'is_weekend'
               ]

"""
"""
dataset = pd.read_csv('OnlineNewsPopularity.csv')

cols = dataset.columns
column_list = ['url']

for i in range(len(cols)-1):
    column_list.append(cols[i+1][1:])

print(column_list)

dataset.columns = column_list

dataset.drop(['url', 'timedelta'], axis=1, inplace=True)

shuffled = dataset.sample(frac=1)

train_data = shuffled[:15000]
test_data = shuffled[15000:]

train_data.to_csv('train.csv', index=False)
test_data.to_csv('test.csv', index=False)
"""

train = pd.read_csv('original.csv', header=None)
print(train)

train_data_cols = list(train.columns)
print(train_data_cols)

train = sample_data(train, 300)
final = generate_MNAR_data(train, q_low=0.6, q_high=1, scale=2, no_y=True)
print(final)
final.to_csv("Facebook_MNAR_60_300_noy.csv", index=None)

