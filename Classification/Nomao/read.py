import numpy as np
import pandas as pd

data = pd.read_csv('train.csv', header=None)

X = data.drop([51, 52, 53, 54, 55, 56], axis=1)
Y = data[[51]]
Y = (Y + 1) / 2


data = pd.read_csv('test.csv', header=None)

print(data.shape)

X_test = data.drop([51, 52, 53, 54, 55, 56], axis=1)
Y_test = data[[56]]
print(Y_test.sum())

