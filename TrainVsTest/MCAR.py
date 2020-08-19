import numpy as np
import random
import pandas as pd


def generate_mcar_data(data, prob=0.5):
    print("shape of data: ", data.shape)
    original_data = data.copy()

    total = 0
    missing_count = 0

    mask_prob = prob * np.ones(shape=data.shape)
    mask = np.ones(shape=data.shape)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):  # column

            total += 1

            if random.random() < mask_prob[i][j]:
                mask[i][j] = 0
                missing_count += 1

    print("percentage of missing values: ", missing_count / total)
    print(mask)

    mask[mask == 0] = 'nan'

    final_data = np.multiply(mask, original_data)

    return final_data


data = pd.read_csv('super_conduct_training_data.csv')
data = data.sample(1000)

data_missing = generate_mcar_data(data, 0.4)
print(data_missing)

data.to_csv('super_conduct_original_1000.csv')
data_missing.to_csv("super_conduct_train_1000_mcar40.csv", index=None)
