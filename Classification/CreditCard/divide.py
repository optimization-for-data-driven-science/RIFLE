import numpy as np
import pandas as pd

data = pd.read_csv('default_credit_cards.csv')

data = data.sample(frac=1)

test = data[:3000]
train = data[3000:]

train.to_csv('train1.csv', index=False)
test.to_csv('test1.csv', index=False)
