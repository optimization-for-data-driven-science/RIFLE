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


x = np.loadtxt('X200K_d50.csv', delimiter=',')
y = np.loadtxt('Y200K_d50.csv', delimiter=',')

y = y[:, np.newaxis]

x_missing = generate_mcar_data(x, 0.6)
y_missing = generate_mcar_data(y, 0.6)

data1 = np.concatenate((x_missing, y_missing), axis=1)
df = pd.DataFrame(data1)

df.to_csv("data_mcar60_d50_200K.csv", index=None)
