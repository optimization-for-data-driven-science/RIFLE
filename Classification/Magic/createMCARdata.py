import pandas as pd
import numpy as np
import random


train = pd.read_csv('train.csv')

missingValueRate = 0.9

n, d = train.shape

mask = np.ones((n, d))

for i in range(n):
    for j in range(d-1):
        t = random.random()
        if t < missingValueRate:
            mask[i][j] = 0

mask[mask == 0] = np.nan

print(mask)
ind = train.index
cols = train.columns
train_missing = np.multiply(mask, train)
print(train_missing)
train_missing.to_csv('train_missing_MCAR90_noy.csv', index=False)
