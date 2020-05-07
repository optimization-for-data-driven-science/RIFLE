import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

# train = pd.read_csv('train_data.csv')
# test = pd.read_csv('test_data.csv')


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

Y = train['10']
X = train.drop(['10'], axis=1)

Y_test = test['10']
X_test = test.drop(['10'], axis=1)


clf = LogisticRegression(random_state=0).fit(X, Y)
print(clf.score(X, Y))
print(clf.score(X_test, Y_test))
