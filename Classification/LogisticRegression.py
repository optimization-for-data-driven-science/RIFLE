import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

"""
data = pd.read_csv('Data_for_UCI_named.csv')


print(data.head())

data.loc[data['stabf'] == 'unstable', 'stabf'] = 0
data.loc[data['stabf'] == 'stable', 'stabf'] = 1
data['Y'] = data['stabf']
data = data.drop(['stabf'], axis=1)

data = data.sample(frac=1).reset_index(drop=True)

train = data[:8000]
test = data[8000:]

train.to_csv('train_data.csv', index=False)
test.to_csv('test_data.csv', index=False)
"""
# train = pd.read_csv('train_data.csv')
# test = pd.read_csv('test_data.csv')

train = pd.read_csv('sonar.all-data', header=None)

train = train.sample(frac=1)

test = train[0: 30]
train = train[30:]

Y = train[60]
X = train.drop([60], axis=1)

Y.loc[Y == 'R'] = 1
Y.loc[Y == 'M'] = 0

Y_test = test[60]
X_test = test.drop([60], axis=1)

Y_test.loc[Y_test == 'R'] = 1
Y_test.loc[Y_test == 'M'] = 0

# Y_test = test['Y']
# X_test = test.drop(['Y'], axis=1)

print(X.shape)
print(Y.shape)

clf = LogisticRegression(random_state=0).fit(X, Y)
print(clf.score(X, Y))
print(clf.score(X_test, Y_test))
