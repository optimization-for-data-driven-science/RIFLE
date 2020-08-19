import random
from statistics import mean
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


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


def generate_MCAR_data(data, prob_vector, no_y=True):
    print("Shape of data: ", data.shape)
    original_data = data.copy()

    cols = data.columns

    total = 0
    missing_count = 0

    mask = np.ones(shape=data.shape)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):  # Column

            total += 1

            if random.random() > prob_vector[j]:
                mask[i][j] = 0
                missing_count += 1

    print("percentage of missing values: ", missing_count / total)
    print(mask)

    mask[mask == 0] = 'nan'

    final_data = np.multiply(mask, original_data.to_numpy())

    pd_data = pd.DataFrame(final_data, columns=cols)
    return pd_data


training_data = read_dataset('training_data.csv')

training_data = sample_data(training_data, 1000)

print(training_data.shape[1])
probabilities = np.random.beta(0.5, 0.5, training_data.shape[1]-1)

for item in probabilities:
    print(item)

probabilities = list(probabilities)

q = 0.25

probabilities = [i * q for i in probabilities]

probabilities.append(0.9)
print("------------")
print(1 - mean(probabilities))
# exit(0)
final = generate_MCAR_data(training_data, prob_vector=probabilities)
training_data.to_csv("original_Beta_MCAR_85_10_1000.csv", index=None)
final.to_csv("missing_Beta_MCAR_85_10_1000.csv", index=None)
