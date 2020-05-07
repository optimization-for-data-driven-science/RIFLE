import pandas as pd

data = pd.read_csv('magic04.data', header=None)
data = data.sample(frac=1)

print(data)

data.loc[data[10] == 'g', 10] = 1
data.loc[data[10] == 'h', 10] = 0

test = data[:3000]
train = data[3000:]

train.to_csv('train.csv', index=False)
test.to_csv('test.csv', index=False)
