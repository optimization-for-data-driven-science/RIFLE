import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

train = pd.read_csv('train1.csv')
test = pd.read_csv('test1.csv')


train = train.fillna(train.mean())

print(train)
Y = train['default payment next month']
X = train.drop(['default payment next month'], axis=1)


Y_test = test['default payment next month']
X_test = test.drop(['default payment next month'], axis=1)

print(X.shape)
print(Y.shape)
clf = LogisticRegression(random_state=0).fit(X, Y)
print(clf.score(X, Y))
print(clf.score(X_test, Y_test))
