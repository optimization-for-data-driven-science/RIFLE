import pandas as pd
from sklearn.preprocessing import StandardScaler
from math import sqrt
import numpy as np
import random


training_data = pd.read_csv('training_data.csv')
test_data = pd.read_csv('test_data.csv')

training_data = training_data.drop(['Po', 'At', 'Rn', 'H', 'F', 'Cl', 'Br', 'I', 'Nd'], axis=1)
# pd.set_option('display.max_rows', -1)


mask_prob = 0.4 * np.ones(training_data.shape)

print(mask_prob)
mask = np.ones(training_data.shape)

count = 0
total = 0

for i in range(training_data.shape[0]):
    for j in range(training_data.shape[1]):
        total += 1
        if random.random() < mask_prob[i][j]:
            mask[i][j] = 0  # Missing
            count += 1

print(mask)
print(count)
print(total)

np.savetxt("mask_pattern_mcar.csv", mask, delimiter=",")

mask[mask == 0] = 'nan'
print(mask)

train = pd.read_csv('training_data.csv')
train = train.drop(['Po', 'At', 'Rn', 'H', 'F', 'Cl', 'Br', 'I', 'Nd'], axis=1)
train_arr = train.to_numpy()


print(mask.shape)
print(train_arr.shape)

final_data = np.multiply(mask, train_arr)

final_pd = pd.DataFrame(final_data, columns=train.columns)
final_pd.to_csv('final_data_missing_mcar.csv', index=False)
