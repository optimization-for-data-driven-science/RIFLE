import pandas as pd
import numpy as np
import random


def generate_MCAR_data(data, prob=0.5):
    print("Shape of data: ", data.shape)
    original_data = data.copy()

    cols = data.columns

    total = 0
    missing_count = 0

    mask_prob = prob * np.ones(shape=data.shape)
    mask = np.ones(shape=data.shape)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):  # Column

            total += 1

            if random.random() < mask_prob[i][j]:
                mask[i][j] = 0
                missing_count += 1

    print("percentage of missing values: ", missing_count / total)
    print(mask)

    mask[mask == 0] = 'nan'

    numpy_version = original_data.to_numpy()
    numpy_version = numpy_version.astype(float)
    print(numpy_version)
    final_data = np.multiply(mask, numpy_version)

    pd_data = pd.DataFrame(final_data, columns=cols)
    return pd_data


dataset = pd.read_csv('Sydney_Data.csv', header=None)

dataset = dataset.sample(400)

final = generate_MCAR_data(dataset, prob=0.50)

dataset.to_csv('original_Sydney_MCAR50_400_5.csv', index=None)
final.to_csv("missing_Sydney_MCAR50_400_5.csv", index=None)
