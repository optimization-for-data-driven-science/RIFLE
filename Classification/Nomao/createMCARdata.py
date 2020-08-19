import pandas as pd
import numpy as np
import random


train = pd.read_csv('train.csv', header=None)

train = train.sample(n=50)
missingValueRate = 0.4

n, d = train.shape

mask = np.ones((n, d))

for i in range(n):
    for j in range(d):
        t = random.random()
        if t < missingValueRate:
            mask[i][j] = 0

mask[mask == 0] = np.nan

print(mask)
ind = train.index
cols = train.columns
train_missing = np.multiply(mask, train)
print(train_missing)
train_missing.to_csv('train_missing_MCAR40_50.csv', index=False, header=None)
