import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


train = pd.read_csv('avila-tr.csv')
test = pd.read_csv('avila-ts.csv')

# Data normalization:
X = train.drop(['11'], axis=1)
Y = train['11']
Y.loc[Y != 'A'] = 0
Y.loc[Y == 'A'] = 1
sc = StandardScaler()

columns = X.columns
ind = X.index

sc.fit(X)
X = sc.transform(X)
X = pd.DataFrame(X, columns=columns, index=ind)

Y_test = test['11']
Y_test.loc[Y_test != 'A'] = 0
Y_test.loc[Y_test == 'A'] = 1

X_test = test.drop(['11'], axis=1)
X_test = sc.transform(X_test)


clf = LogisticRegression(random_state=0, penalty='none').fit(X, Y)
print(clf.score(X, Y))
print(clf.score(X_test, Y_test))
