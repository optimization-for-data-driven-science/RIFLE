import pandas as pd
from sklearn.preprocessing import StandardScaler
from math import sqrt
import numpy as np
import random


training_data = pd.read_csv('training_data.csv')
test_data = pd.read_csv('test_data.csv')

training_data = training_data.sample(500)

train = training_data.copy()

# training_data = training_data.drop(['Po', 'At', 'Rn', 'H', 'F', 'Cl', 'Br', 'I', 'Nd'], axis=1)
# pd.set_option('display.max_rows', -1)

data_array = training_data.to_numpy()

scaler = StandardScaler()

scaler.fit(training_data)

scaled_data = scaler.transform(training_data)

means = scaler.mean_
vars = scaler.var_

stds = []
print(vars)

for item in vars:
    stds.append(sqrt(item))
print(stds)
print(len(stds))
print(scaled_data.shape)

mask_prob = np.zeros(scaled_data.shape)

mask = np.ones(scaled_data.shape)

count1 = 0
count2 = 0
count3 = 0
count4 = 0
count = 0

print(scaled_data.shape)
print(data_array.shape)

p1 = 0.4
p2 = 0.6
p3 = 0.8
p4 = 0.95

for i in range(data_array.shape[0]):
    for j in range(data_array.shape[1]):

        val = data_array[i][j] - means[j]

        if stds[j] > val > - stds[j]:
            mask_prob[i][j] = p1
            count1 += 1

        elif 2 * stds[j] > val > - 2 * stds[j]:
            mask_prob[i][j] = p2
            count2 += 1

        elif 3 * stds[j] > val > - 3 * stds[j]:
            mask_prob[i][j] = p3
            count3 += 1

        else:
            mask_prob[i][j] = p4
            count4 += 1

        if random.random() < mask_prob[i][j]:
            mask[i][j] = 0
            count += 1

print(mask_prob)
total = count1 + count2 + count3 + count4
print(count1 / total)
print(count2 / total)
print(count3 / total)
print(count4 / total)
expectation = count1 * p1 + count2 * p2 + count3 * p3 + count4 * p4
print(expectation)
print(count)
print(expectation / total)
print(count / total)


mask[mask == 0] = 'nan'
print(mask)

# train = pd.read_csv('training_data.csv')
# train = train.drop(['Po', 'At', 'Rn', 'H', 'F', 'Cl', 'Br', 'I', 'Nd'], axis=1)
train_arr = train.to_numpy()


print(mask.shape)
print(train_arr.shape)

final_data = np.multiply(mask, train_arr)

final_pd = pd.DataFrame(final_data, columns=train.columns)
final_pd.to_csv('final_data_missing40n500.csv', index=False)
