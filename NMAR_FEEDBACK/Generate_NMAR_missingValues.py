import pandas as pd
from sklearn.preprocessing import StandardScaler
from math import sqrt, fabs
import numpy as np
import random
import scipy.stats as stat

a = 4
b = 0
val = a * fabs(1) - b
print(stat.norm.cdf(val))

q_list = []

training_data = pd.read_csv('train_data.csv', header=None)

print(training_data)

number_of_features = training_data.shape[1]

for i in range(number_of_features):
    q_list.append((random.random()+1)/2)

test_data = pd.read_csv('test_data.csv')

training_data = training_data.sample(1000)

train = training_data.copy()

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
total = 0
print(scaled_data.shape)
print(data_array.shape)

for i in range(data_array.shape[0]):
    for j in range(data_array.shape[1]):

        total += 1

        val = a * fabs(scaled_data[i][j]) - b

        mask_prob[i][j] = stat.norm.cdf(val) * q_list[j]  # Probability of missing

        if random.random() < mask_prob[i][j]:
            mask[i][j] = 0
            count += 1

"""
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
"""

print(mask_prob)
print(count)
print(count / total)

# exit(0)

mask[mask == 0] = 'nan'
print(mask)

# train = pd.read_csv('training_data.csv')
# train = train.drop(['Po', 'At', 'Rn', 'H', 'F', 'Cl', 'Br', 'I', 'Nd'], axis=1)
train_arr = train.to_numpy()


print(mask.shape)
print(train_arr.shape)

final_data = np.multiply(mask, train_arr)

final_pd = pd.DataFrame(final_data, columns=train.columns)
final_pd.to_csv('feedback_data_missingNMAR50n1K.csv', index=False)
